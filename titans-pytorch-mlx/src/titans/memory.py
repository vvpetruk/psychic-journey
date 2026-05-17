# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Neural Long-term Memory Module for Titans.

This module implements the core innovation of Titans: a neural memory that
learns to memorize at test time using gradient descent with momentum and
weight decay. The memory is trained with an associative memory loss to
learn key-value associations.

Key equations from the paper:
    Memory update: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
    Loss: loss(M; x) = ||M(k) - v||^2

where:
    - alpha_t: forgetting/decay factor (weight decay)
    - eta_t: surprise decay (momentum coefficient)
    - theta_t: learning rate for momentary surprise
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.func import grad, vmap

from titans.config import TitansConfig

# Import optimizations
# Memory Workspace training currently requires the standard PyTorch memory update path.
# The CUDA optimization path compiles on the DGX once Python headers are present,
# but reproduces gradient collapse / uniform-loss behavior in the training loop.
# Keep this disabled until the optimized kernels are validated against the
# standard update numerically and with short training gates.
try:
    from titans.cuda_optimizations import (
        batched_memory_update,
        compute_memory_gradients_efficient,
    )
    HAS_CUDA_OPTIMIZATIONS = False
except ImportError:
    HAS_CUDA_OPTIMIZATIONS = False

# Check for Triton availability
try:
    from titans.triton_kernels import triton_memory_update, is_triton_available
    HAS_TRITON = is_triton_available()
except ImportError:
    HAS_TRITON = False


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def _tensor_norm(value: torch.Tensor) -> float:
    detached = value.detach().float()
    return float(torch.sqrt(torch.sum(detached * detached)).item())


def _tensor_norm_sum(values: list[torch.Tensor]) -> float:
    return sum(_tensor_norm(value) for value in values)


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _expand_update_value(
    value: torch.Tensor | float,
    reference: torch.Tensor,
) -> torch.Tensor | float:
    if not isinstance(value, torch.Tensor):
        return value
    if reference.ndim == 3 and value.ndim == 1:
        return value.view(value.shape[0], 1, 1)
    if reference.ndim == 2 and value.ndim > 0:
        return value.mean()
    return value


def _affine_scan(
    gates: torch.Tensor,
    inputs: torch.Tensor,
    prev: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inclusive scan for h_t = gates_t * h_{t-1} + inputs_t.

    The affine transform composition is associative:
    (a2, b2) o (a1, b1) = (a2 * a1, b2 + a2 * b1).
    Hillis-Steele scan gives us a differentiable segment-wise implementation
    with O(log n) vectorized steps instead of a Python step per token.
    """
    factors = gates
    offsets = inputs
    seq_len = gates.shape[1]
    step = 1

    while step < seq_len:
        next_factors = factors.clone()
        next_offsets = offsets.clone()

        cur_factors = factors[:, step:]
        prev_factors = factors[:, :-step]
        cur_offsets = offsets[:, step:]
        prev_offsets = offsets[:, :-step]

        next_factors[:, step:] = cur_factors * prev_factors
        next_offsets[:, step:] = cur_offsets + cur_factors * prev_offsets

        factors = next_factors
        offsets = next_offsets
        step *= 2

    if prev is None:
        return offsets

    return factors * prev.unsqueeze(1) + offsets


@dataclass
class MemoryState:
    """State of the neural long-term memory.

    This encapsulates the memory weights and momentum for continuing
    inference across chunks/segments.

    Attributes:
        weights: List of weight matrices for each memory layer
        momentum: Accumulated surprise momentum (S_t in paper)
    """

    weights: list[torch.Tensor]
    momentum: list[torch.Tensor]

    def detach(self) -> MemoryState:
        """Detach state from computation graph."""
        return MemoryState(
            weights=[w.detach() for w in self.weights],
            momentum=[m.detach() for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        """Clone the memory state."""
        return MemoryState(
            weights=[w.clone() for w in self.weights],
            momentum=[m.clone() for m in self.momentum],
        )


class MemoryMLP(nn.Module):
    """MLP architecture for the neural memory.

    This is the actual memory module that stores information in its weights.
    It's a simple MLP that learns key-value associations.

    For L_M = 1 (linear memory), this is equivalent to a matrix-valued memory.
    For L_M >= 2 (deep memory), this provides more expressive power.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_memory_layers
        self.dim = config.dim
        self.hidden_dim = config.memory_hidden_dim

        # Build MLP layers
        self.layers = nn.ModuleList()

        if self.num_layers == 1:
            # Linear memory: single linear layer
            self.layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            # Deep memory: MLP with hidden layers
            # First layer: dim -> hidden_dim
            self.layers.append(nn.Linear(self.dim, self.hidden_dim, bias=False))

            # Hidden layers
            for _ in range(self.num_layers - 2):
                self.layers.append(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                )

            # Last layer: hidden_dim -> dim
            self.layers.append(nn.Linear(self.hidden_dim, self.dim, bias=False))

        self.activation = get_activation(config.activation)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values."""
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through memory MLP.

        Args:
            x: Input tensor of shape (batch, seq, dim)

        Returns:
            Output tensor of shape (batch, seq, dim)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Apply activation for all but last layer
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def forward_with_weights(
        self,
        x: torch.Tensor,
        weights: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass using explicit memory weights.

        This keeps the state-to-output path differentiable. Copying state
        weights into ``nn.Parameter.data`` is fine for inference, but it hides
        the updated state from the outer LM loss during training.
        """
        h = x
        for i, weight in enumerate(weights):
            if weight.ndim == 2:
                h = F.linear(h, weight)
            elif weight.ndim == 3:
                h = torch.einsum("bsi,boi->bso", h, weight)
            elif weight.ndim == 4:
                h = torch.einsum("bsi,bsoi->bso", h, weight)
            else:
                raise ValueError(f"Expected memory weight to be 2D, 3D, or 4D, got {weight.ndim}D")
            if i < len(weights) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        """Get current weight matrices."""
        return [layer.weight.data.clone() for layer in self.layers]

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        """Set weight matrices."""
        for layer, w in zip(self.layers, weights, strict=True):
            layer.weight.data.copy_(w)

    def compute_loss(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute associative memory loss.

        Loss: ||M(k) - v||^2

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)

        Returns:
            Scalar loss value
        """
        predictions = self.forward(keys)
        return F.mse_loss(predictions, values, reduction="mean")

    def compute_loss_with_weights(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute associative memory loss with explicit memory weights."""
        predictions = self.forward_with_weights(keys, weights)
        if weights and weights[0].ndim == 3:
            per_sample_loss = (predictions - values).square().mean(dim=(1, 2))
            return per_sample_loss.sum()
        return F.mse_loss(predictions, values, reduction="mean")


class NeuralLongTermMemory(nn.Module):
    """Neural Long-term Memory Module.

    This is the main memory component of Titans. It learns to memorize
    at test time by treating training as an online learning problem.

    The memory is updated using gradient descent with:
    - Momentum (for past surprise)
    - Weight decay (for forgetting)

    Key features:
    1. Data-dependent learning rate, momentum, and decay
    2. Deep memory MLP for expressive power
    3. Surprise-based update rule
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim

        # Projections for keys, values, and queries
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        # Optional 1D depthwise convolution (following Mamba2/GatedDeltaNet)
        self.use_conv = config.use_conv
        if self.use_conv:
            self.conv_k = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_v = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_q = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )

        # The actual memory module
        self.memory = MemoryMLP(config)

        # Data-dependent gates for learning parameters
        # These produce alpha_t (decay), theta_t (lr), eta_t (momentum)
        self.gate_decay = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )
        self.gate_lr = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )
        self.gate_momentum = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.proj_gate = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)
        self.gate_norm = nn.LayerNorm(config.dim)
        self.out_norm = nn.LayerNorm(config.dim)

        self.last_update_stats: dict[str, float] = {}

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [
            self.proj_k,
            self.proj_v,
            self.proj_q,
            self.proj_gate,
            self.proj_out,
        ]:
            nn.init.normal_(module.weight, std=self.config.init_std)

    def _output_transform(
        self,
        retrieved: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the paper's normalized gated readout before output projection."""
        gate = 1.0 + F.silu(self.proj_gate(self.gate_norm(source)))
        return self.proj_out(self.out_norm(retrieved) * gate)

    def _apply_conv(
        self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply 1D convolution to K, V, Q."""
        if not self.use_conv:
            return k, v, q

        # Reshape for conv: (batch, seq, dim) -> (batch, dim, seq)
        k = rearrange(k, "b s d -> b d s")
        v = rearrange(v, "b s d -> b d s")
        q = rearrange(q, "b s d -> b d s")

        # Apply causal convolution
        k = self.conv_k(k)[..., : k.shape[-1]]
        v = self.conv_v(v)[..., : v.shape[-1]]
        q = self.conv_q(q)[..., : q.shape[-1]]

        # Reshape back: (batch, dim, seq) -> (batch, seq, dim)
        k = rearrange(k, "b d s -> b s d")
        v = rearrange(v, "b d s -> b s d")
        q = rearrange(q, "b d s -> b s d")

        return k, v, q

    def _compute_gradients(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Compute gradients for memory update.

        This computes the gradient of the associative memory loss
        with respect to the memory weights.

        Uses optimized gradient computation when available:
        - Analytical gradients for single-layer memory
        - Triton kernels for fused operations
        - Fallback to autograd for complex cases

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)
            weights: Current memory-state weights

        Returns:
            List of gradient tensors for each memory layer
        """
        # Try optimized gradient computation first
        if HAS_CUDA_OPTIMIZATIONS and keys.is_cuda:
            try:
                state_weights = [weight.detach() for weight in weights]
                return compute_memory_gradients_efficient(
                    keys.detach(),
                    values.detach(),
                    state_weights,
                    activation=self.config.activation,
                )
            except Exception:
                pass  # Fall back to standard method

        # Use torch.enable_grad() to compute gradients under torch.no_grad().
        # This is essential because Titans learns at test time. Callers should
        # avoid torch.inference_mode(), which prevents this inner autograd path.
        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            if outer_grad_enabled:
                keys_grad = keys
                values_grad = values
            else:
                keys_grad = keys.detach().requires_grad_(True)
                values_grad = values.detach()

            grad_weights = [
                weight if weight.requires_grad else weight.detach().requires_grad_(True)
                for weight in weights
            ]

            # Compute loss
            loss = self.memory.compute_loss_with_weights(
                keys_grad,
                values_grad,
                grad_weights,
            )

            # Compute gradients. During training, create_graph=True lets the LM
            # loss train the write projections and adaptive gates through the
            # surprise update. During no-grad generation, the inner update still
            # runs but does not retain an outer graph.
            grads = torch.autograd.grad(
                loss,
                grad_weights,
                create_graph=outer_grad_enabled,
                allow_unused=True,
            )

        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, grad_weights, strict=True)
        ]

    def _compute_token_surprises(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Compute per-token associative-loss gradients from segment entry weights."""
        outer_grad_enabled = torch.is_grad_enabled()

        def forward_one(
            token: torch.Tensor,
            token_weights: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            h = token
            for layer_idx, weight in enumerate(token_weights):
                h = F.linear(h, weight)
                if layer_idx < len(token_weights) - 1:
                    h = self.memory.activation(h)
            return h

        def loss_one(
            token_weights: tuple[torch.Tensor, ...],
            key: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            prediction = forward_one(key, token_weights)
            return (prediction - value).square().mean()

        with torch.enable_grad():
            if outer_grad_enabled:
                keys_grad = keys
                values_grad = values
            else:
                keys_grad = keys.detach()
                values_grad = values.detach()

            grad_weights = tuple(
                weight if weight.requires_grad else weight.detach().requires_grad_(True)
                for weight in weights
            )

            try:
                grad_fn = grad(loss_one)
                token_grad_fn = vmap(
                    vmap(grad_fn, in_dims=(None, 0, 0)),
                    in_dims=(0, 0, 0),
                )
                token_grads = token_grad_fn(grad_weights, keys_grad, values_grad)
                return list(token_grads)
            except Exception:
                # Conservative fallback: still compute the chunk's surprises
                # from the entry weights, then the update itself remains scanned.
                per_step = []
                for token_idx in range(keys.shape[1]):
                    per_step.append(
                        self._compute_gradients(
                            keys_grad[:, token_idx : token_idx + 1],
                            values_grad[:, token_idx : token_idx + 1],
                            list(grad_weights),
                        )
                    )
                return [
                    torch.stack([step[layer_idx] for step in per_step], dim=1)
                    for layer_idx in range(len(weights))
                ]

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Initialize memory state.

        Args:
            batch_size: Batch size
            device: Device for tensors

        Returns:
            Initial memory state
        """
        # Each sequence in a batch owns an independent contextual memory state.
        # Preserve the graph during training so the outer LM loss can train the
        # shared memory initialization.
        if torch.is_grad_enabled():
            base_weights = [layer.weight.to(device) for layer in self.memory.layers]
        else:
            base_weights = [weight.to(device) for weight in self.memory.get_weights()]

        weights = [
            weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
            for weight in base_weights
        ]

        # Initialize momentum to zeros
        momentum = [torch.zeros_like(w) for w in weights]

        return MemoryState(weights=weights, momentum=momentum)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        return_state: bool = True,
    ) -> tuple[torch.Tensor, MemoryState | None]:
        """Forward pass with memory update.

        This performs both:
        1. Memory retrieval: query the memory for relevant information
        2. Memory update: update the memory with new key-value pairs

        Args:
            x: Input tensor (batch, seq, dim)
            state: Previous memory state (optional)
            return_state: Whether to return updated state

        Returns:
            Tuple of (output, state) where output is (batch, seq, dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, device)

        # Project to keys, values, queries
        k = self.proj_k(x)
        v = self.proj_v(x)
        q = self.proj_q(x)

        # Apply convolution
        k, v, q = self._apply_conv(k, v, q)

        # Apply SiLU activation (following paper Section 4.4)
        k = F.silu(k)
        v = F.silu(v)
        q = F.silu(q)

        # Normalize queries and keys using L2-norm (Section 4.4)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        pre_weight_norm = _tensor_norm_sum(state.weights)
        pre_momentum_norm = _tensor_norm_sum(state.momentum)

        # In the Titans paper the memory module is recurrent: gates and surprise
        # are functions of the current token, and the state advances at each
        # token. We keep scalar gates per sequence for the current MLP state
        # representation, but avoid the previous batch/chunk-averaged update.
        raw_alpha_steps = self.gate_decay(x).mean(dim=2)
        alpha_steps = torch.clamp(
            raw_alpha_steps * self.config.memory_decay,
            min=0.0,
            max=0.01,
        )
        theta_steps = self.gate_lr(x).mean(dim=2) * self.config.memory_lr
        eta_steps = (
            self.gate_momentum(x).mean(dim=2) * self.config.memory_momentum
        )

        # Compute per-token surprises for the whole segment from the segment
        # entry state, then use associative scans for the recurrent momentum and
        # memory-weight updates. This matches the paper's chunk-wise parallel
        # implementation strategy while preserving per-token writes.
        grads = self._compute_token_surprises(k, v, state.weights)
        grad_norm = _tensor_norm_sum(grads)

        eta_factors = eta_steps[:, :, None, None]
        alpha_factors = 1.0 - alpha_steps[:, :, None, None]
        theta_factors = theta_steps[:, :, None, None]

        momentum_series = []
        weight_series = []
        new_momentum = []
        new_weights = []

        for weight, momentum, grad_series in zip(
            state.weights,
            state.momentum,
            grads,
            strict=True,
        ):
            surprise_inputs = -theta_factors * grad_series
            layer_momentum_series = _affine_scan(
                eta_factors,
                surprise_inputs,
                prev=momentum,
            )
            layer_weight_series = _affine_scan(
                alpha_factors,
                layer_momentum_series,
                prev=weight,
            )
            momentum_series.append(layer_momentum_series)
            weight_series.append(layer_weight_series)
            new_momentum.append(layer_momentum_series[:, -1])
            new_weights.append(layer_weight_series[:, -1])

        retrieved = self.memory.forward_with_weights(q, weight_series)

        weight_delta_norm = sum(
            _tensor_norm(new - old)
            for old, new in zip(state.weights, new_weights, strict=True)
        )
        momentum_delta_norm = sum(
            _tensor_norm(new - old)
            for old, new in zip(state.momentum, new_momentum, strict=True)
        )
        self.last_update_stats = {
            "batch_size": float(batch_size),
            "seq_len": float(seq_len),
            "raw_decay_gate": _scalar(raw_alpha_steps.mean()),
            "raw_decay_gate_min": _scalar(raw_alpha_steps.min()),
            "raw_decay_gate_max": _scalar(raw_alpha_steps.max()),
            "effective_decay": _scalar(alpha_steps.mean()),
            "effective_decay_min": _scalar(alpha_steps.min()),
            "effective_decay_max": _scalar(alpha_steps.max()),
            "effective_lr": _scalar(theta_steps.mean()),
            "effective_lr_min": _scalar(theta_steps.min()),
            "effective_lr_max": _scalar(theta_steps.max()),
            "effective_momentum": _scalar(eta_steps.mean()),
            "effective_momentum_min": _scalar(eta_steps.min()),
            "effective_momentum_max": _scalar(eta_steps.max()),
            "memory_grad_norm_sum": grad_norm,
            "pre_weight_norm_sum": pre_weight_norm,
            "post_weight_norm_sum": _tensor_norm_sum(new_weights),
            "weight_delta_norm_sum": weight_delta_norm,
            "pre_momentum_norm_sum": pre_momentum_norm,
            "post_momentum_norm_sum": _tensor_norm_sum(new_momentum),
            "momentum_delta_norm_sum": momentum_delta_norm,
        }

        output = self._output_transform(retrieved, x)

        # Create new state
        new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        if return_state:
            return output, new_state
        return output, None

    def _standard_memory_update(
        self,
        weights: list[torch.Tensor],
        momentum: list[torch.Tensor],
        grads: list[torch.Tensor],
        alpha: torch.Tensor | float,
        eta: torch.Tensor | float,
        theta: torch.Tensor | float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Standard memory update using PyTorch operations.

        Args:
            weights: Current weight tensors
            momentum: Current momentum tensors
            grads: Gradient tensors
            alpha: Decay factor
            eta: Momentum coefficient
            theta: Learning rate

        Returns:
            Tuple of (new_weights, new_momentum)
        """
        # Update momentum: S_t = eta * S_{t-1} - theta * grad
        new_momentum = []
        for m, g in zip(momentum, grads, strict=True):
            eta_for_m = _expand_update_value(eta, m)
            theta_for_g = _expand_update_value(theta, g)
            s = eta_for_m * m - theta_for_g * g
            new_momentum.append(s)

        # Update weights: M_t = (1 - alpha) * M_{t-1} + S_t
        new_weights = []
        for w, s in zip(weights, new_momentum, strict=True):
            alpha_for_w = _expand_update_value(alpha, w)
            w_new = (1 - alpha_for_w) * w + s
            new_weights.append(w_new)

        return new_weights, new_momentum

    def retrieve(
        self,
        queries: torch.Tensor,
        state: MemoryState,
    ) -> torch.Tensor:
        """Retrieve from memory without updating.

        Args:
            queries: Query vectors (batch, seq, dim)
            state: Memory state to query

        Returns:
            Retrieved values (batch, seq, dim)
        """
        # Project queries
        q = self.proj_q(queries)

        if self.use_conv:
            q = rearrange(q, "b s d -> b d s")
            q = self.conv_q(q)[..., : q.shape[-1]]
            q = rearrange(q, "b d s -> b s d")

        q = F.silu(q)
        q = F.normalize(q, p=2, dim=-1)

        retrieved = self.memory.forward_with_weights(q, state.weights)
        return self._output_transform(retrieved, queries)

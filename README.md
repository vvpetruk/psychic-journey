# Psychic Journey

Public research snapshot for MAC-style Titans neural-memory experiments.

The repository contains:

- a compact experiment workspace with training, evaluation, and diagnostic scripts
- a modified Titans backend with MAC memory update/readout changes
- tests for the memory and model behavior
- an experiment summary with large artifacts omitted

The public snapshot intentionally excludes datasets, checkpoints, run directories, local environment details, credentials, and machine-specific paths.

## Current Focus

- fixed-size neural long-term memory
- per-token surprise-based memory updates
- state carried across stream segments with truncated backpropagation
- boundary-aware language-model training
- instruction-style memory prefill followed by answer scoring/generation
- frozen-answer and live-answer memory-update evaluation modes

## References

- Behrouz, A., Zhong, P., & Mirrokni, V. (2024). [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663). arXiv:2501.00663.
- Original implementation reference: [Aedelon/titans-pytorch-mlx](https://github.com/Aedelon/titans-pytorch-mlx).

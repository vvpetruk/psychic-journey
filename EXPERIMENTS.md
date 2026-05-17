# Experiment Summary

This file records the key MAC/Titans runs represented by this code snapshot. Large artifacts, checkpoints, datasets, local run directories, and machine-specific paths are not committed.

## Boundary-Aware Language-Model Run

Main result:

- best step: `6500`
- best validation loss: `6.039710640907288`
- final validation loss at step `12000`: `6.068546235561371`
- trainable parameters: `7,458,304`

Main hyperparameters:

- steps: `12000`
- batch size: `8`
- gradient accumulation: `1`
- sequence length: `128`
- model dimension: `128`
- layers: `2`
- heads: `4`
- chunk size: `32`
- window size: `32`
- memory decay: `0.001`
- memory learning rate: `0.1`
- memory momentum: `0.9`
- weight decay: `1e-4`
- stream cursor mode: `sharded`
- stateful segments: enabled
- persistent memory: enabled
- persistent eval memory: enabled
- document boundary resets: enabled

Behavioral setup:

- The corpus was treated as continuous token streams.
- Each batch row advanced through its own stream shard.
- Memory state was carried across segments and detached between segments for truncated backpropagation.
- Segments did not cross document boundaries.
- Rows that reached a boundary were reset independently.

## Article Memory Probe

Across 9 article-memory probes from the best checkpoint:

- correct article memory beat reset memory: `5/9`
- correct article memory beat wrong article memory: `2/9`
- mean correct-minus-reset average NLL: `-0.014337`
- mean correct-minus-wrong average NLL: `+0.004495`

Interpretation:

The memory path is active, but the article-specific signal was weak and not reliably better than wrong-article memory.

## Instruction Item-Boundary Fine-Tune

Main setup:

- item-boundary batching
- one independent instruction item per batch row
- prompt/instruction prefill with memory updates enabled
- answer scoring with memory updates frozen for the primary eval
- no sequence crosses from one item into another

Main hyperparameters:

- steps: `3000`
- batch size: `8`
- training episodes: `1000`
- eval episodes: `64`
- learning rate: `1e-4`
- weight decay: `1e-5`
- warmup ratio: `0.05`
- min learning-rate ratio: `0.1`
- gradient clip norm: `1.0`
- max prefill tokens: `192`
- max answer tokens: `128`
- prefill batch tokens: `32`
- persistent memory: disabled for item-boundary run
- write target to memory: disabled

Best frozen-answer eval:

- best step: `800`
- best same-prefill average NLL: `6.781771369278431`

Final frozen-answer eval:

- reset mean average NLL: `7.083300709724426`
- same-prefill average NLL: `7.043424874544144`
- wrong-prefill average NLL: `7.062179744243622`
- same minus reset: `-0.03987583518028259`, wins `59/64`
- same minus wrong: `-0.01875486969947815`, wins `41/64`

## Live Answer Memory-Update Evaluation

This compares frozen answer scoring with teacher-forced answer scoring where memory updates remain enabled for every answer token.

Best checkpoint:

- frozen same-prefill average NLL: `6.781771369278431`
- live same-prefill average NLL: `6.765315517783165`
- live same minus reset: `-0.009791582822799683`, wins `47/64`
- live same minus wrong: `-0.0044347792863845825`, wins `39/64`
- live vs frozen same improvement: `-0.01645585149526596`, wins `49/64`

Final checkpoint:

- frozen same-prefill average NLL: `7.043424874544144`
- live same-prefill average NLL: `7.0260936841368675`
- live same minus reset: `-0.009734414517879486`, wins `41/64`
- live same minus wrong: `-0.008033037185668945`, wins `42/64`
- live vs frozen same improvement: `-0.017331190407276154`, wins `43/64`

Interpretation:

Live test-time answer memory updates improve answer-token NLL slightly over frozen-answer scoring, but the effect remains modest. The next useful research pressure is still on whether the MAC memory is being trained and read strongly enough to produce robust article/task-specific recall.

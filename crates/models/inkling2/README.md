# lattice-model-inkling2

Inkling 2 looped mixture-of-experts text decoder for Lattice, on the CPU

This crate runs the text decoder of the Inkling 2 looped models (`model_type` `inkling2_loop`) on the CPU. A configured run of stored layers is repeated a fixed number of passes for every token. Each pass after the first restarts from the state before the loop and reads messages from the previous pass, and each pass keeps its own key, value and convolution caches.

Experts stay packed in their stored NVFP4 or MXFP8 form and are multiplied by the packed matrix-vector kernels of `lattice-weights`. Other matrices keep their stored element type and are widened to f32 as they are read. Arithmetic is f32 throughout.

`Inkling2Model::open` reads a checkpoint directory whose `config.json` is validated by `lattice-arch`. `Inkling2CpuDecoder` pairs the model with its tokenizer and generates through the shared decoder driver in `lattice-arch`; `lattice-serve` uses it to serve these checkpoints on the CPU. Only the text decoder is loaded: tensors outside it, such as the vision tower's, are ignored. Multi-token prediction and Metal execution are not implemented.

The design is recorded in [ADR-106](../../../docs/adr/ADR-106-inkling2-loop-model.md).

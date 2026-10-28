# lattice-router

Request-level LoRA adapter routing for Lattice. A router gate scores the adapters available to a
request and picks the ones it serves with, and how much weight each one gets. This crate holds the
gate and the pieces that keep it honest across restarts:

- `state`: the on-disk gate artifact, versioned by a counter and a content hash, carrying the
  adapter names the gate was trained on so a wrong pairing is refused instead of served.
- `init`: provisioning a router state directory with a deterministic, untrained bootstrap gate.
- `episode`: recorded routing decisions and the feedback that arrived for them.
- `mixture` (feature `mixture`): the `lattice-fann` backed routing module.
- `refit` and `admission` (feature `router-learning`): bounded refits from recorded feedback, and
  held-out admission evidence for a refit candidate before it may serve.

`lattice-inference` re-exports these modules at their pre-split paths
(`lattice_inference::router_state`, `router_init`, `router_episode`, `router_refit`,
`router_admission` and `mixture`) for code written before the split; those paths are compatibility
re-exports and will be removed at 1.0.

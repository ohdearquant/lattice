# lattice-forecast

Native time-series forecasting for Lattice. Reverso returns point forecasts of a univariate
history; Chronos-2 and Toto 2.0 return quantile forecasts and can forecast related series jointly.
Reverso's context preparation (missing values, padding, striding, resampling) lives in `prep` so it
matches its reference evaluation; Chronos-2 and Toto 2.0 prepare their own input inside the model.

Weights load from safetensors through `lattice-weights`, the CPU kernels come from
`lattice-compute`, and errors use the shared `lattice-inference-core` type. `lattice-inference`
re-exports this crate at `lattice_inference::forecast` for code written before the split; that path
is a compatibility re-export and will be removed at 1.0.

The `forecast_chronos2_goldens`, `forecast_reverso_goldens`, and `forecast_toto2_goldens` integration
tests compare the respective CPU models against external reference fixtures. Set the model-specific
fixture-directory environment variable named in each test before running it; otherwise it skips.

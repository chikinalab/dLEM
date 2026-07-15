# Python API Reference

`import dlem` exposes a single flat namespace; internals live in submodules (`io`,
`model`, `losses`, `train`, `metrics`, `viz`).

## I/O (`dlem.io`)

::: io.load_band

::: io.load_patch

::: io.load_ctcf

::: io.normalize_oe

::: io.extract_band_from_mcool

## Model (`dlem.model`)

::: model.full_rollout

::: model.full_rollout_selfnorm

::: model.teacher_forced

## Losses (`dlem.losses`)

::: losses.make_losses

::: losses.band_metrics

## Fitting (`dlem.train`)

::: train.fit

::: train.DlemFit

::: train.fit_lean

::: train.run_adam

## Metrics (`dlem.metrics`)

::: metrics.ctcf_report

::: metrics.lagged_corr

::: metrics.max_lagged_correlation

::: metrics.track_correlations

::: metrics.load_ctcf_tsv

## Visualization (`dlem.viz`)

::: viz.plot_patch

::: viz.plot_prediction

::: viz.plot_prediction_region

::: viz.band_to_dense

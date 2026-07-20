"""Optimization: Adam with (mse, corr, loss) checkpoints + early stopping, and fit wrappers.

`fit_lean` / `fit` train L, R directly with Adam under the per-diagonal multinomial loss on
the self-normalizing full rollout, trained end-to-end -- loss-neutral under the O/E losses
but stable to full depth.

Checkpoint metrics are always computed on the *plain full rollout* (deployment forward), so
the reported mse/corr reflect deployment behavior.
"""
from dataclasses import dataclass, field

import numpy as np

import jax
import jax.numpy as jnp
import optax

from .model import full_rollout, full_rollout_selfnorm
from .losses import band_metrics, make_losses
from .io import normalize_oe

_INIT = {"mse": np.inf, "loss": np.inf, "corr": -np.inf}
_BETTER = {"mse": lambda a, b: a < b, "loss": lambda a, b: a < b, "corr": lambda a, b: a > b}


def _corr_lr(L, R, max_lag=10):
    """Max *positive* lagged Pearson corr between L and R (within-fit L/R coupling; drift proxy).

    High (near +1) => L and R stay coupled (barriers symmetric); low/negative => the fit has
    drifted L and R apart. (Max positive lagged Pearson corr of L vs R.)
    """
    a = np.asarray(L, np.float32); b = np.asarray(R, np.float32)
    m = min(len(a), len(b)); a, b = a[:m], b[:m]
    best = -2.0
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a[lag:], b[:-lag]) if lag > 0 else (a[:lag], b[-lag:]) if lag < 0 else (a, b)
        if len(aa) < 2:
            continue
        c = np.corrcoef(aa, bb)[0, 1]
        if np.isfinite(c) and c > best:
            best = c
    return float(best)


def run_adam(
    params,
    loss_fn,
    metric_forward,
    obs,
    proj,
    *,
    n_opt=500,
    lr=1e-2,
    every=25,
    start_row=5,
    monitor="mse",
    patience=None,
    verbose=True,
    lr_of=None,
    vel_of=None,
    mse_tol=1.05,
    corr_lr_max_lag=10,
    elbow_smooth=None,
):
    """Adam loop tracking best-mse / best-corr / best-loss params, plus two interpretable checkpoints.

    monitor selects which checkpoint drives early stopping and which params to return.
    patience = number of *checkpoints* with no improvement before stopping (None = no early stop).

    Interpretability checkpoints (barrier tracks stay L/R-coupled / CTCF-aligned):
      * **"symm"** (needs `lr_of`): params with **max corr(L,R) among checkpoints whose deployment
        mse <= mse_tol * best_mse** (least-drifted fit within `mse_tol` of best mse).
      * **"symm2"** (needs `lr_of` AND `vel_of`): the **velocity-elbow** checkpoint -- where the Adam
        parameter MOTION hands off from the symmetric (Sigma) subspace to the antisymmetric (Delta)
        subspace, i.e. the first step where the smoothed Delta-motion fraction fD =
        ||u_antisym|| / (||u_sym|| + ||u_antisym||) crosses 0.5. Parameter-free (no mse budget);
        empirically coincides with `symm` across H1/K562 x 10kb/1kb at ~80-110 steps. `vel_of` maps
        the optax updates pytree to (u_sym, u_antisym); the uniform (loss-invisible) mode is removed.

    Returns {metric: (value, params)} + "history"; also "symm"/"symm_meta" and, with `vel_of`,
    "symm2"/"symm2_meta" (= step, mse, corr_lr, elbow_step).
    """
    if monitor in ("symm", "symm2") and lr_of is None:
        raise ValueError(f"monitor={monitor!r} requires lr_of (a params -> (L, R) function).")
    if monitor == "symm2" and vel_of is None:
        raise ValueError("monitor='symm2' requires vel_of (updates -> (u_sym, u_antisym)).")
    opt = optax.adam(lr)
    state = opt.init(params)

    if vel_of is None:
        @jax.jit
        def step(params, state):
            loss, g = jax.value_and_grad(loss_fn)(params)
            updates, state = opt.update(g, state, params)
            params = proj(optax.apply_updates(params, updates))
            return params, state, loss, 0.0, 0.0
    else:
        @jax.jit
        def step(params, state):
            loss, g = jax.value_and_grad(loss_fn)(params)
            updates, state = opt.update(g, state, params)
            uS, uD = vel_of(updates)
            vS = jnp.linalg.norm(uS - uS.mean())
            vD = jnp.linalg.norm(uD - uD.mean())
            params = proj(optax.apply_updates(params, updates))
            return params, state, loss, vS, vD

    best = {m: (_INIT[m], params) for m in _INIT}
    history = []
    traj = []       # (step, mse, corr_lr, host-copy params) -- only when lr_of given
    vel_hist = []   # per-step (||u_sym||, ||u_antisym||) -- only when vel_of given
    since_improve = 0
    for i in range(n_opt):
        params, state, loss, vS, vD = step(params, state)
        if vel_of is not None:
            vel_hist.append((float(vS), float(vD)))
        if i % every == 0 or i == n_opt - 1:
            pred = metric_forward(params)
            mse, corr = band_metrics(pred, obs, start_row)
            vals = {"mse": mse, "corr": corr, "loss": float(loss)}
            improved_monitor = False
            for m in _INIT:
                if _BETTER[m](vals[m], best[m][0]):
                    best[m] = (vals[m], params)
                    if m == monitor:
                        improved_monitor = True
            clr = None
            if lr_of is not None:
                L, R = lr_of(params)
                clr = _corr_lr(L, R, corr_lr_max_lag)
                traj.append((i, mse, clr, {k: np.asarray(v) for k, v in params.items()}))
            history.append((i, mse, corr, float(loss)))
            if verbose:
                extra = f"  corr(L,R)={clr:+.4f}" if clr is not None else ""
                print(f"  [{i:4d}] loss={float(loss):.4f}  mse={mse:.4f}  corr={corr:.4f}{extra}", flush=True)
            # Early stopping only for the simple monotone monitors (mse/corr/loss).
            if monitor in _INIT:
                since_improve = 0 if improved_monitor else since_improve + 1
                if patience is not None and since_improve >= patience:
                    if verbose:
                        print(f"  early stop @ {i} (no {monitor} improvement in {patience} checks)", flush=True)
                    break
    if traj:
        best_mse = min(t[1] for t in traj)
        thr = mse_tol * best_mse
        eligible = [t for t in traj if t[1] <= thr]
        step_i, mse_i, clr_i, params_i = max(eligible, key=lambda t: t[2])
        best["symm"] = (clr_i, params_i)
        best["symm_meta"] = (step_i, mse_i, clr_i)
    if traj and vel_hist:
        vs = np.array([v[0] for v in vel_hist]); vd = np.array([v[1] for v in vel_hist])
        fD = vd / (vs + vd + 1e-12)
        w = elbow_smooth or max(every, 11)
        fDs = np.convolve(fD, np.ones(w) / w, mode="same") if len(fD) >= w else fD
        cross = np.where(fDs >= 0.5)[0]
        elbow = int(cross[0]) if len(cross) else int(np.argmax(fDs))
        j = min(range(len(traj)), key=lambda t: abs(traj[t][0] - elbow))
        s_j, m_j, c_j, p_j = traj[j]
        best["symm2"] = (c_j, p_j)
        best["symm2_meta"] = (s_j, m_j, c_j, elbow)
    best["history"] = history
    return best


# --------------------------------------------------------------------------------------
# Fit wrappers
# --------------------------------------------------------------------------------------
def _proj_LR(p):
    return {**p, "L": jnp.clip(p["L"], 1e-4, 1.0), "R": jnp.clip(p["R"], 1e-4, 1.0)}


def _optimize(band, *, s, train_to, start_row, importance_power, diag_weight_power,
              n_opt, lr, every, monitor, mse_tol, patience, verbose):
    """Run the Adam optimization; return run_adam's full `best` dict (all checkpoints)."""
    obs = jnp.asarray(band[:train_to], jnp.float32)
    n = obs.shape[1]
    imp, _ = make_losses(obs, start_row, importance_power, diag_weight_power)
    params = {"L": jnp.ones(n), "R": jnp.ones(n)}
    LR_of = lambda p: (p["L"], p["R"])
    vel_of = lambda u: ((u["L"] + u["R"]) / 2.0, (u["L"] - u["R"]) / 2.0)
    loss_fn = lambda p: imp(full_rollout_selfnorm(*LR_of(p), s, train_to))
    metric_forward = lambda p: full_rollout(*LR_of(p), s, train_to)
    return run_adam(params, loss_fn, metric_forward, obs, _proj_LR, n_opt=n_opt, lr=lr,
                    every=every, start_row=start_row, monitor=monitor, patience=patience,
                    verbose=verbose, lr_of=LR_of, vel_of=vel_of, mse_tol=mse_tol)


def fit_lean(
    band,
    *,
    s,
    train_to,
    start_row=5,
    importance_power=0.0,
    diag_weight_power=1.0,
    n_opt=500,
    lr=1e-2,
    every=25,
    monitor="symm2",
    mse_tol=1.05,
    patience=None,
    verbose=True,
):
    """Fit dLEM L, R directly with Adam under the per-diagonal multinomial loss on the
    self-normalizing full rollout, trained end-to-end.

    importance_power  -- WITHIN each diagonal: emphasize high-mass anchor bins (obs distribution ^
                         (1 + importance_power)). 0 = plain multinomial.
    diag_weight_power -- BETWEEN diagonals: weight each diagonal by (diagonal total)^power.
                         1 = mass-weighted; 0 = every diagonal equal.
    monitor (which checkpoint to return; see run_adam):
      "symm2" -- DEFAULT, the velocity-elbow (interpretable, CTCF-aligned) fit.
      "symm"  -- max corr(L,R) among checkpoints within mse_tol x best mse.
      "mse" / "corr" / "loss" -- best-of-that-metric (most-fit but usually drifted).
    Returns (L, R) at the chosen checkpoint.
    """
    best = _optimize(band, s=s, train_to=train_to, start_row=start_row,
                     importance_power=importance_power, diag_weight_power=diag_weight_power,
                     n_opt=n_opt, lr=lr, every=every, monitor=monitor,
                     mse_tol=mse_tol, patience=patience, verbose=verbose)
    bp = best[monitor][1]
    return jnp.clip(bp["L"], 1e-4, 1.0), jnp.clip(bp["R"], 1e-4, 1.0)


# --------------------------------------------------------------------------------------
# High-level result object + fit()
# --------------------------------------------------------------------------------------
@dataclass
class DlemFit:
    """Result of dlem.fit: the barrier tracks L, R (at the selected `monitor` checkpoint) plus
    everything needed to predict/plot.

    `checkpoints` maps each checkpoint name ('mse','corr','loss','symm','symm2') to its (L, R),
    all from the SAME optimization -- pass the DlemFit to dlem.ctcf_report to compare CTCF alignment
    across checkpoints without re-fitting. predict(rows) -> full-rollout band; oe(rows) -> its
    log-O/E; barrier() -> (1-L)(1-R).  mse/corr are the selected checkpoint's deployment O/E metrics.
    """
    L: np.ndarray
    R: np.ndarray
    s: float
    train_to: int
    monitor: str = "symm2"
    importance_power: float = 0.0
    diag_weight_power: float = 1.0
    mse: float = None
    corr: float = None
    checkpoints: dict = field(default_factory=dict)   # name -> (L, R) for every checkpoint

    def _LR(self, checkpoint=None):
        if checkpoint is not None and self.checkpoints and checkpoint in self.checkpoints:
            return self.checkpoints[checkpoint]
        return self.L, self.R

    def predict(self, rows=None, checkpoint=None):
        """Full-rollout band (rows diagonals) from `checkpoint`'s L/R (default the selected one)."""
        L, R = self._LR(checkpoint)
        rows = int(rows or self.train_to)
        L = jnp.asarray(L); R = jnp.asarray(R)
        return np.asarray(full_rollout(L, R, self.s, rows))

    def oe(self, rows=None, checkpoint=None):
        return np.asarray(normalize_oe(self.predict(rows, checkpoint)))

    def barrier(self, checkpoint=None):
        L, R = self._LR(checkpoint)
        return (1.0 - np.asarray(L)) * (1.0 - np.asarray(R))


def fit(band, *, s, train_to, importance_power=0.0, diag_weight_power=1.0, monitor="symm2",
        n_opt=1500, lr=1e-2, every=25, start_row=5, verbose=False):
    """Fit the dLEM and return a DlemFit. The single supported method: optimize L, R directly with
    Adam under the per-diagonal multinomial loss on the self-normalizing full rollout.

    importance_power  -- WITHIN each diagonal: emphasize high-mass anchor bins (0 = plain multinomial).
    diag_weight_power -- BETWEEN diagonals: weight by (diagonal total)^power (1 = mass-weighted, 0 = equal).
    monitor           -- checkpoint returned as fit.L/fit.R: "symm2" (default) | "symm" | "mse" |
                         "corr" | "loss". ALL checkpoints from this one optimization are also kept in
                         fit.checkpoints (compare them with dlem.ctcf_report(fit, ...)).
    """
    best = _optimize(band, s=s, train_to=train_to, start_row=start_row,
                     importance_power=importance_power, diag_weight_power=diag_weight_power,
                     n_opt=n_opt, lr=lr, every=every, monitor=monitor,
                     mse_tol=1.05, patience=None, verbose=verbose)

    def _LR(name):
        p = best[name][1]
        return (np.asarray(jnp.clip(p["L"], 1e-4, 1.0)), np.asarray(jnp.clip(p["R"], 1e-4, 1.0)))

    checkpoints = {name: _LR(name) for name in ("mse", "corr", "loss", "symm", "symm2") if name in best}
    L, R = checkpoints[monitor]
    obs = jnp.asarray(band[:train_to], jnp.float32)
    pred = full_rollout(jnp.asarray(L), jnp.asarray(R), s, train_to)
    mse, corr = band_metrics(pred, obs, start_row)
    return DlemFit(L=L, R=R, s=float(s), train_to=int(train_to), monitor=monitor,
                   importance_power=float(importance_power), diag_weight_power=float(diag_weight_power),
                   mse=float(mse), corr=float(corr), checkpoints=checkpoints)

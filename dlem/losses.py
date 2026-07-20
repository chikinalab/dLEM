"""Training losses and checkpoint metrics.

Both losses are per-diagonal scale-invariant (they normalize each diagonal), which is what
lets the self-normalizing full rollout be loss-neutral.

  * multinomial (`imp_loss`): treat each diagonal as a distribution over anchor bins j; minimize
    the cross-entropy of the observed distribution under the predicted one. Two orthogonal knobs:

      - `importance_power` (WITHIN a diagonal): emphasize high-mass anchor bins. The observed
        per-diagonal distribution is raised to the power (1 + importance_power) and renormalized,
        so 0 = plain multinomial and >0 concentrates the loss on the strongest entries.
      - `diag_weight_power` (BETWEEN diagonals): weight each diagonal's contribution by
        (diagonal total)^diag_weight_power, normalized over valid diagonals. 0 = every diagonal
        equal; 1 = weighted by observed mass.
  * oe-mse (`mse_loss`): MSE between log-O/E(pred) and log-O/E(obs) over the valid region.

`band_metrics` returns pooled (mse, corr) on log-O/E over the valid region, for checkpointing.
"""
import jax.numpy as jnp

from .io import normalize_oe

EPS = 1e-8


def _valid_mask(obs, start_row):
    width, n = obs.shape
    col = jnp.arange(n)[None, :]
    row = jnp.arange(width)[:, None]
    return (col >= row) & (obs > 0) & (row >= start_row)


def make_losses(obs, start_row=5, importance_power=0.0, diag_weight_power=1.0):
    """Return (multinomial_loss, mse_loss) closures over a fixed observed band.

    multinomial_loss(pred): per-diagonal multinomial cross-entropy. `importance_power` emphasizes
        high-mass anchor bins WITHIN each diagonal (0 = plain multinomial);
        `diag_weight_power` weights BETWEEN diagonals by (diagonal total)^power (1 = mass-weighted).
    mse_loss(pred): MSE of log-O/E over the valid region.
    """
    obs = jnp.asarray(obs, jnp.float32)
    mask = _valid_mask(obs, start_row).astype(jnp.float32)
    T = obs * mask
    RT = T.sum(-1, keepdims=True)  # per-diagonal observed total

    # WITHIN-diagonal target distribution, emphasized by importance_power:
    # emphasized_dist ∝ observed_prob^(1 + importance_power), renormalized per diagonal.
    RP = jnp.where(RT > 0, T / jnp.maximum(RT, EPS), 0.0)  # observed per-diagonal distribution
    emph = jnp.power(RP + EPS, 1.0 + importance_power) * mask
    Q = jnp.where(RT > 0, emph / jnp.maximum(emph.sum(-1, keepdims=True), EPS), 0.0)  # target dist

    # BETWEEN-diagonal weights: (diagonal total)^diag_weight_power over valid diagonals.
    row_valid = (mask.sum(-1) > 0).astype(obs.dtype)
    w = jnp.power(jnp.maximum(RT[:, 0], 0.0), diag_weight_power) * row_valid
    w = w / jnp.maximum(w.sum(), EPS)

    obs_oe = normalize_oe(obs) * mask

    def multinomial_loss(pred):
        Pm = pred * mask
        PP = Pm / jnp.maximum(Pm.sum(-1, keepdims=True), EPS)
        ce = -(Q * jnp.log(PP + EPS)).sum(-1)  # per diagonal cross-entropy
        return (w * ce).sum()

    def mse_loss(pred):
        pred_oe = normalize_oe(pred) * mask
        return jnp.sum((pred_oe - obs_oe) ** 2) / jnp.maximum(mask.sum(), EPS)

    return multinomial_loss, mse_loss


def band_metrics(pred, obs, start_row=5):
    """Pooled (mse, corr) on log-O/E over the valid region (numpy floats)."""
    obs = jnp.asarray(obs, jnp.float32)
    mask = _valid_mask(obs, start_row)
    po = normalize_oe(pred)[mask]
    oo = normalize_oe(obs)[mask]
    mse = float(jnp.mean((po - oo) ** 2))
    pc = po - po.mean()
    oc = oo - oo.mean()
    denom = jnp.sqrt(jnp.sum(pc ** 2) * jnp.sum(oc ** 2)) + EPS
    corr = float(jnp.sum(pc * oc) / denom)
    return mse, corr

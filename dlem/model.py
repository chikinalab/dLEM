"""dLEM forward operators.

The dLEM is a recursive band operator: diagonal k is computed from diagonal k-1,
    A[k, j] = (R[j-1]*A[k-1, j-1] + L[i+1]*A[k-1, j]) / (R[j] + L[i] + s),   i = j - k.

L[i] is the probability that the LEFT anchor (currently at site i) extrudes one further
step left, to i-1; R[j] is the probability that the RIGHT anchor (currently at site j)
extrudes one further step right, to j+1 -- i.e. each array is indexed by, and governs the
movement of, the anchor it's named for. (A prior version of this code had L and R's roles
swapped internally -- L was driving the right-moving anchor and R the left-moving one, the
opposite of their names -- fixed here; see also CTCF strand convention in metrics.py.)

Three forwards, by use:
  * `teacher_forced`  -- shallow, data-seeded refine (~10-30 sweeps). Trainable at any depth
                         because the recursion is only `steps` deep. Used for L/R fits.
  * `full_rollout`    -- deep rollout from the constant diagonal (deployment / eval). Not
                         renormalized; forward-only.
  * `full_rollout_selfnorm` -- deep rollout that renormalizes each diagonal to mean 1 during
                         the recursion. For a scale-free (O/E) loss this is loss-neutral
                         (the per-diagonal scale is a free gauge), but it prevents the
                         underflow/floor-clipping that destroys deep gradients -- so the full
                         rollout itself becomes trainable in O(band) memory.
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax


def _clip_LR(L, R):
    return jnp.clip(L, 1e-4, 1.0), jnp.clip(R, 1e-4, 1.0)


def roll_right_1(v):
    return jnp.concatenate([v[-1:], v[:-1]], 0)


# --------------------------------------------------------------------------------------
# Full rollout (deployment / eval) -- plain, forward-only
# --------------------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("rows",))
def full_rollout(L, R, s, rows):
    """Generate the whole band from a constant (ones) diagonal."""
    L, R = _clip_LR(L, R)
    n = L.shape[0]
    init = jnp.ones(n, L.dtype)

    def body(carry, _):
        prev, Lc = carry
        p1, p2 = prev[:-1], prev[1:]
        # R: right anchor, static column index j.  L: left anchor, rolled with k (see
        # module docstring for the derivation of which role gets which treatment).
        r_in, r_out = R[:-1], R[1:]
        l_in, l_out = Lc[1:], Lc[:-1]
        num = r_in * p1 + l_in * p2
        den = jnp.maximum(r_out + l_out + s, 1e-4)
        nxt = prev.at[1:].set(jnp.maximum(num / den, 1e-6)).at[0].set(1.0)
        return (nxt, roll_right_1(Lc)), nxt

    (_, _), tail = lax.scan(body, (init, L), xs=None, length=rows - 1)
    return jnp.vstack([init[None], tail])


# --------------------------------------------------------------------------------------
# Self-normalizing full rollout -- trainable full rollout
# --------------------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("rows",))
def full_rollout_selfnorm(L, R, s, rows):
    """Full rollout that renormalizes each diagonal to mean 1 (over j >= k) in-recursion.

    Loss-neutral for a per-diagonal-normalized (O/E / multinomial) loss, but numerically
    stable to full depth, so it can be trained with plain `jax.grad` in O(band) memory.
    The scan body is gradient-checkpointed (jax.checkpoint) so backprop stays O(band) even
    at deep rollouts (e.g. 1kb, rows~1000); without it the reverse pass OOMs.
    """
    L, R = _clip_LR(L, R)
    n = L.shape[0]
    cols = jnp.arange(n)
    init = jnp.ones(n, L.dtype)

    def body(carry, k):
        prev, Lc = carry
        p1, p2 = prev[:-1], prev[1:]
        r_in, r_out = R[:-1], R[1:]
        l_in, l_out = Lc[1:], Lc[:-1]
        num = r_in * p1 + l_in * p2
        den = jnp.maximum(r_out + l_out + s, 1e-4)
        nxt = prev.at[1:].set(jnp.maximum(num / den, 1e-6)).at[0].set(1.0)
        valid = cols >= k
        cnt = jnp.maximum(jnp.sum(valid.astype(nxt.dtype)), 1.0)
        m = jnp.sum(jnp.where(valid, nxt, 0.0)) / cnt
        nxt = jnp.where(valid, nxt / jnp.maximum(m, 1e-12), nxt)
        return (nxt, roll_right_1(Lc)), nxt

    (_, _), tail = lax.scan(jax.checkpoint(body), (init, L), xs=jnp.arange(1, rows))
    return jnp.vstack([init[None], tail])


# --------------------------------------------------------------------------------------
# Teacher-forced shallow refine (training)
# --------------------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("steps", "start_row"))
def teacher_forced(obs, L, R, s, steps, start_row):
    """Seed from the observed band and apply the recursion `steps` times.

    Per-sweep gradient checkpointing keeps memory O(band) for deep/many-step fits.
    """
    L, R = _clip_LR(L, R)
    D = obs.shape[0]
    rolls = vmap_roll(L, D)  # rolls[t] = roll(L, t) -- left anchor, rolled with k
    upd = jnp.arange(D)[:, None] >= start_row

    def sweep(band, _):
        prev = band[:-1]
        Lk = rolls[: D - 1]
        p1, p2 = prev[:, :-1], prev[:, 1:]
        r_in, r_out = R[:-1], R[1:]
        l_in, l_out = Lk[:, 1:], Lk[:, :-1]
        num = r_in * p1 + l_in * p2
        den = jnp.maximum(r_out + l_out + s, 1e-4)
        new = jnp.maximum(num / den, 1e-6)
        rows1 = jnp.concatenate([jnp.ones((D - 1, 1), band.dtype), new], 1)
        cand = jnp.concatenate([band[:1], rows1], 0)
        out = jnp.where(upd, cand, band).at[:, 0].set(1.0)
        return out, None

    final, _ = lax.scan(jax.checkpoint(sweep), obs, xs=None, length=steps)
    return final


def vmap_roll(v, D):
    return jax.vmap(lambda sh: jnp.roll(v, sh))(jnp.arange(D))

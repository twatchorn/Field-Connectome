"""Field-coupled neural simulation with emergent connectome recording.

No pre-built synaptic weight matrix. Coupling between neurons is through a
Gaussian field deposit: when neuron i fires, the local membrane potential at
neuron j is raised by K[i,j] = exp(-d_ij²/2λ²) (signed by i's excitatory/
inhibitory polarity). The firing probability rule is unchanged from the
original model: P(fire | V) = h + (1-h)·exp(β·(V - V_th)).

Two emergent connectomes are recorded in parallel:
  - C_lenient: every (i fires at t-1, j fires at t) pair gets credit K[i,j].
               Multiple contributors share credit, weighted by field strength.
  - C_strict:  only the single largest contributor to j's field kick gets
               credit (1 per causal event). Directed, sparse, less noisy.

Both are stored as sparse matrices that accumulate over the run.

Usage:
    python field_connectome.py --out OUTDIR --N 20 --beta 0.6
                               [--lambda 3.0] [--steps 10000]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


# =============================================================================
# Parameters
# =============================================================================

# Neuron dynamics (consistent with NeuralProp_MCMC.py)
V_rest, V_th, V_reset = -70.0, -55.0, -75.0
tau_mem  = 20.0
sigma    = 1.0
t_ref    = 5
p_exc    = 0.8
h        = 1e-4

# Field coupling
w_0_default     = 2.5    # peak field deposit (V kick) from a fired neighbor at d=0
lambda_default  = 3.0    # Gaussian length scale (lattice units)
kernel_cutoff   = 3.0    # truncate Gaussian beyond this many λ (3σ → ~99% mass)


# =============================================================================
# Field kernel
# =============================================================================

def lattice_coords(N):
    return np.indices((N, N, N)).reshape(3, -1).T


def build_kernel(N, lam, w_0, cutoff_in_lambdas=3.0):
    """Build the sparse Gaussian field-coupling kernel.

    K[i,j] = w_0 * exp(-d_ij² / (2 λ²))  for d_ij ≤ cutoff_in_lambdas * λ,
    zero elsewhere. Self-coupling (i==j) is excluded.

    Returns
    -------
    K      : (n_nodes, n_nodes) CSR sparse matrix
    coords : (n_nodes, 3) int array of positions
    """
    coords = lattice_coords(N)
    n_nodes = len(coords)
    cutoff = cutoff_in_lambdas * lam
    r_int  = int(np.ceil(cutoff))

    rows, cols, vals = [], [], []
    for i, c in enumerate(coords):
        # bounding-box prefilter
        in_box = np.all(np.abs(coords - c) <= r_int, axis=1)
        cand = np.where(in_box)[0]
        diffs = coords[cand] - c
        d2 = np.sum(diffs ** 2, axis=1)
        mask = (d2 > 0) & (d2 <= cutoff ** 2)
        targets = cand[mask]
        weights = w_0 * np.exp(-d2[mask] / (2 * lam ** 2))
        rows.extend([i] * len(targets))
        cols.extend(targets.tolist())
        vals.extend(weights.tolist())

    return (sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes),
                          dtype=np.float32),
            coords)


# =============================================================================
# Simulation
# =============================================================================

def run_field_sim(K, n_steps, beta, neuron_type, rng,
                  V_init=None, refractory_init=None,
                  track_connectomes=True, record_activity=True):
    """Run field-coupled simulation, recording both connectomes if requested.

    Returns dict with:
      C_lenient, C_strict : sparse accumulator matrices
      activity_ts, spikes_ts : per-step time series
      V, refractory : final state (for chaining rounds)
    """
    n = K.shape[0]

    V          = V_init if V_init is not None else (
                 V_rest + 5.0 * rng.standard_normal(n).astype(np.float32))
    refractory = refractory_init if refractory_init is not None else (
                 np.zeros(n, dtype=np.int32))

    activity_ts = np.empty(n_steps) if record_activity else None
    spikes_ts   = np.empty(n_steps) if record_activity else None

    # Connectome accumulators: keep COO lists, sparsify at end
    C_len_rows, C_len_cols, C_len_vals = [], [], []
    C_str_rows, C_str_cols, C_str_vals = [], [], []

    fired_prev = np.zeros(n, dtype=bool)
    K_T = K.T.tocsr()    # we'll need K^T for the matvec (K[i,j] deposits to j)

    # Periodic flush parameters — accumulate up to ~1M entries before merging
    # into a running CSR. Prevents OOM on long runs with high activity.
    FLUSH_THRESHOLD = 1_000_000
    C_lenient_running = sp.csr_matrix((n, n), dtype=np.float32)
    C_strict_running  = sp.csr_matrix((n, n), dtype=np.float32)

    def _flush():
        nonlocal C_lenient_running, C_strict_running
        nonlocal C_len_rows, C_len_cols, C_len_vals
        nonlocal C_str_rows, C_str_cols, C_str_vals
        if C_len_rows:
            tmp = sp.coo_matrix((C_len_vals, (C_len_rows, C_len_cols)),
                                shape=(n, n), dtype=np.float32).tocsr()
            C_lenient_running = C_lenient_running + tmp
            C_len_rows, C_len_cols, C_len_vals = [], [], []
        if C_str_rows:
            tmp = sp.coo_matrix((C_str_vals, (C_str_rows, C_str_cols)),
                                shape=(n, n), dtype=np.float32).tocsr()
            C_strict_running = C_strict_running + tmp
            C_str_rows, C_str_cols, C_str_vals = [], [], []

    for t in range(n_steps):
        # membrane update — same as NeuralProp_MCMC
        V += -(V - V_rest) / tau_mem + sigma * rng.standard_normal(n).astype(np.float32)

        # field deposit from current firing (will apply BEFORE next step's threshold check)
        active = refractory <= 0
        p_intrinsic = np.where(V > V_th, 1.0, np.exp(beta * (V - V_th)))
        prob   = h + (1.0 - h) * p_intrinsic
        fired  = (rng.random(n) < prob) & active

        # ---- record connectome credit BEFORE applying the new field deposit ----
        # Credit goes to (i fired at t-1, j fires at t) pairs:
        #   - lenient: credit = K[i,j] for every such (i,j)
        #   - strict:  credit = 1 only if i was the largest contributor to j
        if track_connectomes and fired_prev.any() and fired.any():
            i_idx = np.where(fired_prev)[0]
            j_idx = np.where(fired)[0]
            # Pull out the (n_i, n_j) sub-matrix as dense — small in practice
            # since typically n_i, n_j << n_nodes.
            K_sub = K[i_idx][:, j_idx]
            if K_sub.nnz > 0:
                # Lenient credit: every nonzero in K_sub is an event
                coo = K_sub.tocoo()
                C_len_rows.extend(i_idx[coo.row].tolist())
                C_len_cols.extend(j_idx[coo.col].tolist())
                C_len_vals.extend(coo.data.tolist())

                # Strict credit: argmax along axis 0 for each column j.
                # Dense conversion is fine here because the sub-matrix is small.
                K_sub_dense = K_sub.toarray()
                # For columns with at least one positive entry, find row of max
                col_has_entry = (K_sub_dense > 0).any(axis=0)
                if col_has_entry.any():
                    best_rows = K_sub_dense.argmax(axis=0)
                    valid_cols = np.where(col_has_entry)[0]
                    C_str_rows.extend(i_idx[best_rows[valid_cols]].tolist())
                    C_str_cols.extend(j_idx[valid_cols].tolist())
                    C_str_vals.extend([1.0] * len(valid_cols))

        # ---- apply field deposit from this step's firing to V ----
        # The kick affects V for the next step (so it shows up as field_prev next iter)
        if fired.any():
            kick = K_T @ (fired.astype(np.float32) * neuron_type)
            V += kick

        V[fired]          = V_reset
        refractory[fired] = t_ref
        refractory[refractory > 0] -= 1

        if record_activity:
            activity_ts[t] = fired.mean()
            spikes_ts[t]   = fired.sum()

        fired_prev = fired.copy()

        # periodic flush to prevent OOM — check every step (cheap)
        if track_connectomes and (
            len(C_len_rows) > FLUSH_THRESHOLD or len(C_str_rows) > FLUSH_THRESHOLD
        ):
            _flush()

    # final flush
    if track_connectomes:
        _flush()
    C_lenient = C_lenient_running
    C_strict  = C_strict_running

    return {
        "C_lenient":    C_lenient,
        "C_strict":     C_strict,
        "activity_ts":  activity_ts,
        "spikes_ts":    spikes_ts,
        "V":            V,
        "refractory":   refractory,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="./field_out")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--lam", type=float, default=lambda_default,
                        help="Gaussian length scale λ (lattice units)")
    parser.add_argument("--w0", type=float, default=w_0_default,
                        help="peak field kick at d=0")
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--burn-in", type=int, default=1000,
                        help="steps to run with connectome tracking off")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {outdir}")

    rng = np.random.default_rng(args.seed)

    # ---- build field kernel ----
    print(f"\nBuilding Gaussian field kernel: N={args.N}, λ={args.lam}, w₀={args.w0}")
    t0 = time.time()
    K, coords = build_kernel(args.N, args.lam, args.w0)
    n = K.shape[0]
    print(f"  n_nodes = {n}, kernel nnz = {K.nnz} "
          f"({K.nnz / n:.1f} interactions per neuron), "
          f"build took {time.time()-t0:.1f}s")
    print(f"  K stats: min={K.data.min():.4f}  max={K.data.max():.4f}  "
          f"mean={K.data.mean():.4f}")

    # ---- initial state ----
    V          = V_rest + 5.0 * rng.standard_normal(n).astype(np.float32)
    refractory = np.zeros(n, dtype=np.int32)
    neuron_type = rng.choice([1, -1], size=n, p=[p_exc, 1 - p_exc]).astype(np.float32)

    # ---- burn-in (no connectome tracking) ----
    if args.burn_in > 0:
        print(f"\nBurn-in: {args.burn_in} steps...")
        t0 = time.time()
        bi = run_field_sim(K, args.burn_in, args.beta, neuron_type, rng,
                           V_init=V, refractory_init=refractory,
                           track_connectomes=False, record_activity=False)
        V, refractory = bi["V"], bi["refractory"]
        print(f"  burn-in took {time.time()-t0:.1f}s")

    # ---- main recording run ----
    print(f"\nRecording: {args.steps} steps at β={args.beta}")
    t0 = time.time()
    out = run_field_sim(K, args.steps, args.beta, neuron_type, rng,
                        V_init=V, refractory_init=refractory,
                        track_connectomes=True, record_activity=True)
    print(f"  recording took {time.time()-t0:.1f}s")
    print(f"  mean activity: {out['activity_ts'].mean():.5f}")
    print(f"  total spikes:  {int(out['spikes_ts'].sum())}")
    print(f"  C_lenient: nnz={out['C_lenient'].nnz}, "
          f"sum={out['C_lenient'].sum():.2f}, max={out['C_lenient'].max():.4f}")
    print(f"  C_strict:  nnz={out['C_strict'].nnz}, "
          f"sum={int(out['C_strict'].sum())}, max={int(out['C_strict'].max())}")

    # ---- save ----
    print("\nSaving outputs...")
    sp.save_npz(outdir / "K.npz", K)
    sp.save_npz(outdir / "C_lenient.npz", out["C_lenient"])
    sp.save_npz(outdir / "C_strict.npz", out["C_strict"])
    np.savez(outdir / "diagnostics.npz",
             beta=args.beta, lam=args.lam, w0=args.w0,
             N=args.N, steps=args.steps,
             coords=coords,
             neuron_type=neuron_type,
             activity_ts=out["activity_ts"],
             spikes_ts=out["spikes_ts"])
    print(f"  K.npz, C_lenient.npz, C_strict.npz, diagnostics.npz → {outdir}")


if __name__ == "__main__":
    main()

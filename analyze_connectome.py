"""Analyze field-connectome simulation outputs.

Loads K.npz, C_lenient.npz, C_strict.npz, and diagnostics.npz from an output
directory and produces a battery of analysis plots:

  01_activity_trace.png         — activity time series, burn-in already removed
  02_weight_distributions.png   — log-log P(w) for K, C_lenient, C_strict
  03_degree_distributions.png   — in/out degree distributions (loglog)
  04_strength_distributions.png — total in/out weight per neuron
  05_spatial_decay.png          — edge weight vs Euclidean distance
  06_lenient_vs_strict.png      — scatter of C_lenient[i,j] vs C_strict[i,j]
                                  on shared support
  07_connectome_summary.txt     — text summary of key statistics

Usage:
    python analyze_connectome.py --in OUTDIR [--threshold-pct 90]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as sp


def _save(fig, outdir, name, dpi=150):
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def log_hist(x, n_bins=30):
    """Log-spaced histogram. Returns bin centers, density."""
    x = np.asarray(x)
    x = x[x > 0]
    if len(x) < 10:
        return None, None
    bins = np.logspace(np.log10(x.min()), np.log10(x.max() + 1e-12), n_bins)
    h, e = np.histogram(x, bins=bins, density=True)
    c = 0.5 * (e[:-1] + e[1:])
    m = h > 0
    return c[m], h[m]


def edge_distance_pairs(C, coords, max_pairs=200000):
    """Return (distance, weight) pairs for edges in C. Subsample if too many."""
    coo = C.tocoo()
    if coo.nnz == 0:
        return np.array([]), np.array([])
    rows, cols, vals = coo.row, coo.col, coo.data
    if len(rows) > max_pairs:
        idx = np.random.default_rng(0).choice(len(rows), size=max_pairs, replace=False)
        rows, cols, vals = rows[idx], cols[idx], vals[idx]
    d = np.linalg.norm(coords[rows] - coords[cols], axis=1)
    return d, vals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="indir", type=str, required=True,
                        help="input directory containing the .npz files")
    parser.add_argument("--out", type=str, default=None,
                        help="output directory for plots (default: same as input)")
    parser.add_argument("--threshold-pct", type=float, default=90.0,
                        help="percentile threshold for 'strong edges' analysis")
    args = parser.parse_args()

    indir  = Path(args.indir).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve() if args.out else indir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Input  ← {indir}")
    print(f"Output → {outdir}\n")

    # -------- load --------
    K          = sp.load_npz(indir / "K.npz")
    C_lenient  = sp.load_npz(indir / "C_lenient.npz")
    C_strict   = sp.load_npz(indir / "C_strict.npz")
    diag       = np.load(indir / "diagnostics.npz")
    coords     = diag["coords"]
    activity   = diag["activity_ts"]
    spikes     = diag["spikes_ts"]
    beta       = float(diag["beta"])
    lam        = float(diag["lam"])
    w0         = float(diag["w0"])
    N          = int(diag["N"])
    n_nodes    = K.shape[0]

    print(f"β = {beta}, λ = {lam}, w₀ = {w0}, N = {N}, n_nodes = {n_nodes}")
    print(f"K:         nnz = {K.nnz:>10}  ({K.nnz/n_nodes:.1f}/neuron)")
    print(f"C_lenient: nnz = {C_lenient.nnz:>10}  ({C_lenient.nnz/n_nodes:.1f}/neuron)")
    print(f"C_strict:  nnz = {C_strict.nnz:>10}  ({C_strict.nnz/n_nodes:.1f}/neuron)")
    print()

    print("Generating figures...")

    # -------- 01: activity trace --------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1.plot(activity, lw=0.5)
    ax1.set_ylabel("activity (frac firing)")
    ax1.set_title(f"Activity time series (β={beta}, λ={lam}, w₀={w0})")
    ax2.plot(spikes, lw=0.5, color="C1")
    ax2.set_xlabel("step")
    ax2.set_ylabel("total spikes / step")
    fig.tight_layout()
    _save(fig, outdir, "01_activity_trace")

    # -------- 02: weight distributions (loglog P(w)) --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, M, name in [
        (axes[0], K,         "K (kernel)"),
        (axes[1], C_lenient, "C_lenient"),
        (axes[2], C_strict,  "C_strict"),
    ]:
        if M.nnz == 0:
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center")
            ax.set_title(name)
            continue
        xs, ys = log_hist(M.data, n_bins=30)
        if xs is not None:
            ax.loglog(xs, ys, 'o-', ms=3, lw=1)
        ax.set_xlabel("edge weight")
        ax.set_ylabel("P(w)")
        ax.set_title(f"{name} (nnz={M.nnz})")
    fig.tight_layout()
    _save(fig, outdir, "02_weight_distributions")

    # -------- 03: degree distributions --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for M, name, color in [
        (C_lenient, "C_lenient", "C0"),
        (C_strict,  "C_strict",  "C1"),
    ]:
        if M.nnz == 0:
            continue
        in_deg  = np.asarray((M != 0).sum(axis=0)).ravel()
        out_deg = np.asarray((M != 0).sum(axis=1)).ravel()
        for ax, deg, kind in [(axes[0], in_deg, "in"), (axes[1], out_deg, "out")]:
            xs, ys = log_hist(deg, n_bins=25)
            if xs is not None:
                ax.loglog(xs, ys, 'o-', ms=3, lw=1, color=color,
                          label=f"{name} (⟨k⟩={deg.mean():.1f})")
    axes[0].set(xlabel="in-degree", ylabel="P(k_in)", title="In-degree")
    axes[1].set(xlabel="out-degree", ylabel="P(k_out)", title="Out-degree")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, outdir, "03_degree_distributions")

    # -------- 04: total strength per neuron --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for M, name, color in [
        (C_lenient, "C_lenient", "C0"),
        (C_strict,  "C_strict",  "C1"),
    ]:
        if M.nnz == 0:
            continue
        in_str  = np.asarray(M.sum(axis=0)).ravel()
        out_str = np.asarray(M.sum(axis=1)).ravel()
        for ax, s, kind in [(axes[0], in_str, "in"), (axes[1], out_str, "out")]:
            xs, ys = log_hist(s, n_bins=25)
            if xs is not None:
                ax.loglog(xs, ys, 'o-', ms=3, lw=1, color=color,
                          label=f"{name} (⟨s⟩={s.mean():.2f})")
    axes[0].set(xlabel="incoming strength", ylabel="P(s_in)",
                title="Total incoming weight")
    axes[1].set(xlabel="outgoing strength", ylabel="P(s_out)",
                title="Total outgoing weight")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, outdir, "04_strength_distributions")

    # -------- 05: spatial decay (weight vs distance) --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, M, name in [
        (axes[0], K,         "K (theoretical Gaussian)"),
        (axes[1], C_lenient, "C_lenient (observed)"),
        (axes[2], C_strict,  "C_strict (observed)"),
    ]:
        d, v = edge_distance_pairs(M, coords)
        if len(d) == 0:
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center")
            ax.set_title(name)
            continue
        # bin by distance — vectorized with bincount
        d_max = d.max()
        n_bins = 25
        bins   = np.linspace(0, d_max, n_bins + 1)
        idx    = np.clip(np.digitize(d, bins) - 1, 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)
        sums   = np.bincount(idx, weights=v, minlength=n_bins)
        sumsq  = np.bincount(idx, weights=v ** 2, minlength=n_bins)
        valid  = counts > 0
        means  = np.where(valid, sums / np.maximum(counts, 1), np.nan)
        # standard error of the mean
        vars_  = np.where(valid, sumsq / np.maximum(counts, 1) - means ** 2, np.nan)
        errs   = np.where(valid, np.sqrt(np.maximum(vars_, 0) / np.maximum(counts, 1)), np.nan)
        bin_d  = 0.5 * (bins[:-1] + bins[1:])
        ax.errorbar(bin_d[valid], means[valid], yerr=errs[valid],
                    fmt='o-', ms=4, capsize=3)
        # theoretical Gaussian overlay for K
        if name.startswith("K"):
            d_theory = np.linspace(0.5, d_max, 100)
            ax.plot(d_theory, w0 * np.exp(-d_theory ** 2 / (2 * lam ** 2)),
                    'k--', alpha=0.5, label="w₀·exp(-d²/2λ²)")
            ax.legend(fontsize=8)
        ax.set_xlabel("distance d")
        ax.set_ylabel("mean weight")
        ax.set_title(name)
        ax.set_yscale("log")
    fig.tight_layout()
    _save(fig, outdir, "05_spatial_decay")

    # -------- 06: lenient vs strict on shared support --------
    fig, ax = plt.subplots(figsize=(6, 6))
    if C_lenient.nnz > 0 and C_strict.nnz > 0:
        # C_strict is (by construction) a subset of C_lenient's support — every
        # strict event is also a lenient event. So we just need C_lenient's
        # value at every nonzero of C_strict.
        strict_coo = C_strict.tocoo()
        rows = strict_coo.row
        cols = strict_coo.col
        x_strict = strict_coo.data

        # batched CSR lookup of lenient[rows, cols]
        # scipy returns a 1xN matrix; squeeze to 1D
        lenient_csr = C_lenient.tocsr()
        y_lenient = np.asarray(lenient_csr[rows, cols]).ravel()

        # subsample if there are too many points to render usefully
        MAX_PTS = 100_000
        if len(x_strict) > MAX_PTS:
            sub = np.random.default_rng(0).choice(len(x_strict), MAX_PTS, replace=False)
            x_strict = x_strict[sub]
            y_lenient = y_lenient[sub]
            ax.set_title(f"Lenient vs strict on shared edges (subsampled to {MAX_PTS:,})")
        else:
            ax.set_title("Lenient vs strict credit on shared edges")

        m = (x_strict > 0) & (y_lenient > 0)
        if m.any():
            ax.loglog(x_strict[m], y_lenient[m], '.', ms=2, alpha=0.3)
            ax.set_xlabel("C_strict[i,j] (causal events)")
            ax.set_ylabel("C_lenient[i,j] (weighted credit)")
            # diagonal reference
            lo = min(x_strict[m].min(), y_lenient[m].min())
            hi = max(x_strict[m].max(), y_lenient[m].max())
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, label="y=x")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "empty connectome", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    _save(fig, outdir, "06_lenient_vs_strict")

    # -------- 07: text summary --------
    summary_path = outdir / "07_connectome_summary.txt"
    threshold_pct = args.threshold_pct
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Connectome analysis summary\n")
        f.write(f"# Input: {indir}\n\n")
        f.write(f"Parameters: β={beta}, λ={lam}, w₀={w0}, N={N}, n_nodes={n_nodes}\n\n")

        f.write(f"## Activity\n")
        f.write(f"  mean activity:  {activity.mean():.5f}\n")
        f.write(f"  std activity:   {activity.std():.5f}\n")
        f.write(f"  max activity:   {activity.max():.5f}\n")
        f.write(f"  total spikes:   {int(spikes.sum())}\n\n")

        for M, name in [(K, "K (kernel)"),
                        (C_lenient, "C_lenient"),
                        (C_strict, "C_strict")]:
            f.write(f"## {name}\n")
            f.write(f"  nnz:          {M.nnz}\n")
            if M.nnz == 0:
                f.write("  (empty)\n\n")
                continue
            f.write(f"  sum:          {M.sum():.4f}\n")
            f.write(f"  mean weight:  {M.data.mean():.4f}\n")
            f.write(f"  max weight:   {M.data.max():.4f}\n")
            in_deg  = np.asarray((M != 0).sum(axis=0)).ravel()
            out_deg = np.asarray((M != 0).sum(axis=1)).ravel()
            f.write(f"  mean in-deg:  {in_deg.mean():.2f}\n")
            f.write(f"  mean out-deg: {out_deg.mean():.2f}\n")
            f.write(f"  max in-deg:   {in_deg.max()}\n")
            f.write(f"  max out-deg:  {out_deg.max()}\n")
            # what fraction of edges carry top-X% of weight?
            data_sorted = np.sort(M.data)[::-1]
            cum = np.cumsum(data_sorted)
            frac_carrying_top = np.searchsorted(cum, threshold_pct / 100 * M.sum())
            f.write(f"  edges carrying {threshold_pct:.0f}% of total weight: "
                    f"{frac_carrying_top} ({100 * frac_carrying_top / M.nnz:.1f}%)\n\n")

        f.write(f"## Lenient vs strict overlap\n")
        if C_lenient.nnz > 0 and C_strict.nnz > 0:
            # Vectorized set arithmetic via packed (row, col) integer keys.
            # n_nodes < 2^32 so a single int64 holds the encoded pair safely.
            len_r, len_c = C_lenient.nonzero()
            str_r, str_c = C_strict.nonzero()
            len_keys = len_r.astype(np.int64) * n_nodes + len_c
            str_keys = str_r.astype(np.int64) * n_nodes + str_c
            both = np.intersect1d(len_keys, str_keys, assume_unique=True)
            f.write(f"  edges in both:        {len(both)}\n")
            f.write(f"  in lenient only:      {len(len_keys) - len(both)}\n")
            f.write(f"  in strict only:       {len(str_keys) - len(both)}\n")
            f.write(f"  fraction of strict that's in lenient: "
                    f"{len(both)/max(len(str_keys),1):.3f}\n")
        else:
            f.write("  (one or both empty)\n")

    print(f"  saved → {summary_path.name}")
    print(f"\nDone. All outputs in {outdir}")


if __name__ == "__main__":
    main()

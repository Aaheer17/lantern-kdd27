#!/usr/bin/env python3
# usage:
#   python plot_training_csv.py --csv /path/to/training_log.csv --smooth 200 --clip 5.0 --bins 100
#
# Creates:
#   fig1_loss_total.png
#   fig2_decomposition.png
#   fig3_weights.png
#   fig4_r_eff_timeseries.png
#   fig5_r_eff_hist.png

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _rolling(y, w):
    """Centered moving average (handles NaNs)."""
    if w is None or w <= 1:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    return pd.Series(y).rolling(window=w, center=True, min_periods=max(1, w//4)).mean().to_numpy()

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _safe_div(num, den):
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out
def plot_fig7_step_vs_mse_and_cfd(train_csv_path: str, out_path: str):
    steps, loss_param, cfd_metric = [], [], []
    print("zakaria pocha")
    with open(train_csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            s  = row.get("opt_step") or row.get("global_step") or row.get("step")
            lp = row.get("loss_param")
            cm = row.get("CFD_metric")

            try:
                s  = float(s)  if s not in (None, "") else np.nan
                lp = float(lp) if lp not in (None, "") else np.nan
                cm = float(cm) if cm not in (None, "") else np.nan
            except Exception:
                s, lp, cm = np.nan, np.nan, np.nan

            steps.append(s)
            loss_param.append(lp)
            cfd_metric.append(cm)

    steps = np.asarray(steps, dtype=float)
    loss_param = np.asarray(loss_param, dtype=float)
    cfd_metric = np.asarray(cfd_metric, dtype=float)

    m = np.isfinite(steps) & np.isfinite(loss_param) & np.isfinite(cfd_metric)
    if m.sum() == 0:
        print("[warn] Figure 7: no finite (opt_step, loss_param, CFD_metric) rows to plot.")
        return

    steps, loss_param, cfd_metric = steps[m], loss_param[m], cfd_metric[m]

    plt.figure()
    plt.plot(steps, loss_param, label="loss_param")
    plt.plot(steps, cfd_metric, label="CFD_metric")
    plt.xlabel("Optimizer step")
    plt.ylabel("Value")
    plt.title("Optimization Step vs MSE loss & CFD_metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] wrote {out_path}")

def main(csv_path, smooth, r_clip, bins):
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist
    cols = [
        "loss_total","loss_param","CFD_metric","mse_noise","mse_x0","mse_v",
            "cfd_raw","w_t_mean","lambda_warm","lambda_cfd_eff",
            "cfd_fro_pred","cfd_fro_shrunk","cfd_fro_smooth","cfd_ema_dist","raw_mvn","energy_loss"
    ]
    # --- NEW: ensure Frobenius diagnostic columns exist and are numeric ---
    fro_cols = ["cfd_fro_pred", "cfd_fro_shrunk", "cfd_fro_smooth", "cfd_ema_dist"]
    for c in fro_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = _to_num(df[c])
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = _to_num(df[c])

    # Drop rows without opt_step or loss
    df = df.dropna(subset=["opt_step", "loss_total"])
    df = df.sort_values("opt_step").reset_index(drop=True)

    # --- r_eff = (lambda_cfd_eff * cfd_raw) / mse_noise ---
    if "lambda_cfd_eff" not in df.columns:
        df["lambda_cfd_eff"] = np.nan
    # compute with guards
    num = (df["lambda_cfd_eff"] * df["cfd_raw"]).to_numpy(dtype=float)
    den = df["mse_noise"].to_numpy(dtype=float)
    r_eff = _safe_div(num, den)
    # optional clipping (just for plotting readability)
    if r_clip is not None and r_clip > 0:
        r_plot = np.clip(r_eff, -r_clip, r_clip)
    else:
        r_plot = r_eff.copy()
    df["r_eff"] = r_eff

    x = df["opt_step"].to_numpy()
    loss_param=df['loss_param'].to_numpy()
    CFD_metric=df['CFD_metric'].to_numpy()
    mvn_loss = df["mvn_loss"].to_numpy()
    energy_loss=df["energy_loss"].to_numpy()
    y_loss = df["loss_total"].to_numpy()
    y_loss_s = _rolling(y_loss, smooth)
    raw_mvn= df["raw_mvn"].to_numpy()

    # === Figure 1: total loss ===
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y_loss, linewidth=1, alpha=0.35, label="loss_total (raw)")
    if smooth and smooth > 1:
        plt.plot(x, y_loss_s, linewidth=2, label=f"loss_total (rolling {smooth})")
    plt.xlabel("optimizer step")
    plt.ylabel("loss_total")
    plt.title("Training Loss per Optimizer Step")
    plt.legend()
    out1 = os.path.join(os.path.dirname(csv_path), "fig1_loss_total.png")
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.show()

    # === Figure 2: decomposition ===
    have_mse = df["mse_noise"].notna().any()
    have_cfd = df["cfd_raw"].notna().any()
    have_eff = df["lambda_cfd_eff"].notna().any()

    plt.figure(figsize=(8, 4.5))
    if have_mse:
        plt.plot(x, _rolling(df["mse_noise"].to_numpy(), smooth), label="mse_noise")
    if have_cfd:
        plt.plot(x, _rolling(df["cfd_raw"].to_numpy(), smooth), label="cfd_raw")
    ax = plt.gca()
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss components")
    title = "Loss Decomposition"
    if have_eff:
        ax2 = ax.twinx()
        ax2.plot(x, _rolling(df["lambda_cfd_eff"].to_numpy(), smooth), linestyle="--", label="lambda_cfd_eff")
        ax2.set_ylabel("lambda_cfd_eff")
        title += " (+ lambda_cfd_eff)"
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")
    plt.title(title)
    out2 = os.path.join(os.path.dirname(csv_path), "fig2_decomposition.png")
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.show()

    # === Figure 3: weights context ===
    have_wt = df["w_t_mean"].notna().any()
    have_warm = df["lambda_warm"].notna().any()
    if have_wt or have_warm:
        plt.figure(figsize=(8, 4.5))
        if have_wt:
            plt.plot(x, _rolling(df["w_t_mean"].to_numpy(), smooth), label="w_t_mean")
        if have_warm:
            plt.plot(x, _rolling(df["lambda_warm"].to_numpy(), smooth), label="lambda_warm")
        plt.xlabel("optimizer step")
        plt.ylabel("value")
        plt.title("Weights Context (w_t_mean, lambda_warm)")
        plt.legend(loc="best")
        out3 = os.path.join(os.path.dirname(csv_path), "fig3_weights.png")
        plt.tight_layout(); plt.savefig(out3, dpi=150); plt.show()
    else:
        out3 = None

    # === Figure 4: r_eff time series ===
    have_reff = np.isfinite(r_eff).any()
    if have_reff:
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, _rolling(r_plot, smooth), label=f"r_eff (rolling {smooth})" if smooth and smooth>1 else "r_eff", linewidth=2)
        plt.xlabel("optimizer step")
        plt.ylabel("r_eff = (lambda_cfd_eff * cfd_raw) / mse_noise")
        ttl = "Effective CFD-to-MSE Ratio (r_eff)"
        if r_clip is not None and r_clip > 0:
            ttl += f"  [clipped to ±{r_clip}]"
        plt.title(ttl)
        plt.legend(loc="best")
        out4 = os.path.join(os.path.dirname(csv_path), "fig4_r_eff_timeseries.png")
        plt.tight_layout(); plt.savefig(out4, dpi=150); plt.show()
    else:
        out4 = None

    # === Figure 5: r_eff histogram (distribution) ===
    if have_reff:
        plt.figure(figsize=(8, 4.5))
        # use unclipped values for stats but clip for display if requested
        vals = r_eff.copy()
        vals_disp = r_plot.copy()
        vals = vals[np.isfinite(vals)]
        vals_disp = vals_disp[np.isfinite(vals_disp)]
        if len(vals) > 0:
            mu = np.nanmean(vals)
            p95 = np.nanpercentile(np.abs(vals), 95)
            plt.hist(vals_disp, bins=bins, density=True, alpha=0.8)
            plt.axvline(0.0, color='k', linewidth=1)
            plt.title(f"r_eff distribution (n={len(vals)}), mean={mu:.3g}, abs-95%={p95:.3g}"
                      + (f", clipped ±{r_clip}" if r_clip else ""))
            plt.xlabel("r_eff"); plt.ylabel("density")
            out5 = os.path.join(os.path.dirname(csv_path), "fig5_r_eff_hist.png")
            plt.tight_layout(); plt.savefig(out5, dpi=150); plt.show()
        else:
            out5 = None
    else:
        out5 = None
        
    # === Figure 6: Frobenius diagnostics (pred → shrunk → smooth; ema_dist dashed) ===
    def _num(s):
    # coerce strings like "" to NaN; tensors/objects to float if possible
        return pd.to_numeric(s, errors="coerce")

    fro_cols = ["cfd_fro_pred","cfd_fro_shrunk","cfd_fro_smooth","cfd_ema_dist"]
    df[fro_cols] = df[fro_cols].apply(_num)
    
    have_fro_any = df[fro_cols].notna().any().any()
    
    if have_fro_any:
        plt.figure(figsize=(8, 4.5))
    
        def plot_one(col, label, ls="-"):
            s = _num(df[col])
            m = s.notna()
            if m.any():
                x_i = df.loc[m, "opt_step"].to_numpy() if "opt_step" in df else np.arange(m.sum())
                y_i = _rolling(s.loc[m].to_numpy(), smooth)
                # align x with y if your _rolling drops the first (smooth-1) points
                if len(y_i) != len(x_i):
                    # assume your _rolling returns same length with NaNs at the start; mask them out
                    v = ~np.isnan(y_i)
                    x_i, y_i = x_i[v], y_i[v]
                plt.plot(x_i, y_i, label=label, linestyle=ls)
    
        plot_one("cfd_fro_pred",   "fro_pred",   "-")
        plot_one("cfd_fro_shrunk", "fro_shrunk", "-")
        plot_one("cfd_fro_smooth", "fro_smooth", "-")
        plot_one("cfd_ema_dist",   "ema_dist",   "--")
    
        plt.xlabel("optimizer step"); plt.ylabel("off-diag Frobenius")
        plt.title("CFD Frobenius diagnostics (pred → shrunk → smooth; ema_dist dashed)")
        plt.legend(loc="best")
        out6 = os.path.join(os.path.dirname(csv_path), "fig6_fro_diagnostics.png")
        plt.tight_layout(); plt.savefig(out6, dpi=150); plt.show()
    else:
        out6 = None
    #print("why zakaria is pocha?")
    out7 = os.path.join(os.path.dirname(csv_path), "fig7_step_lossparam_vs_CFDmetric.png")
    fig, ax1 = plt.subplots(figsize=(6,4))
    #plot_fig7_step_vs_mse_and_cfd(csv_path, out7)
    c1 = 'tab:blue'
    c2 = 'tab:orange'
    #c3 = 'tab:green'
    # First series on left y-axis
    ln1 = ax1.plot(x, loss_param,  linewidth=1.5,
                   label='loss_param', color=c1)[0]
    ax1.set_xlabel('opt_step')
    ax1.set_ylabel('MSE loss', color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    
    # Second series on right y-axis
    ax2 = ax1.twinx()
    ln2 = ax2.plot(x, CFD_metric, linewidth=1.5,
                   label='CFD_metric',  color=c2)[0]
    ax2.set_ylabel('CFD_metric', color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    
    # One combined legend
    lines = [ln1, ln2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    #plt.show()
    plt.savefig(out7,dpi=150)

    out_mvn = os.path.join(os.path.dirname(csv_path), "fig_factored_mvn_loss_vs_opt_step.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(x, mvn_loss, linewidth=1.5, label="mvn_loss")
    # ax.plot(x, y_smooth, linewidth=1.5, linestyle="--", label="mvn_loss (MA-25)")  # optional
    
    ax.set_xlabel("opt_step")
    ax.set_ylabel("factored_mvn_loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    fig.tight_layout()
    plt.savefig(out_mvn, dpi=150)
    # plt.show()
    print(f"Saved: {out_mvn}")

    out_mvn_raw = os.path.join(os.path.dirname(csv_path), "fig_raw_mvn_loss_vs_opt_step.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(x, raw_mvn, linewidth=1.5, label="raw_mvn_loss")
    # ax.plot(x, y_smooth, linewidth=1.5, linestyle="--", label="mvn_loss (MA-25)")  # optional
    
    ax.set_xlabel("opt_step")
    ax.set_ylabel("raw_mvn_loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    fig.tight_layout()
    plt.savefig(out_mvn_raw, dpi=150)
    # plt.show()
    print(f"Saved: {out_mvn}")
    
    out_energy_loss=os.path.join(os.path.dirname(csv_path), "fig_energy_loss_vs_opt_step.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, energy_loss, linewidth=1.5, label="energy_loss")
    # ax.plot(x, y_smooth, linewidth=1.5, linestyle="--", label="mvn_loss (MA-25)")  # optional
    
    ax.set_xlabel("opt_step")
    ax.set_ylabel("energy_loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    fig.tight_layout()
    plt.savefig(out_energy_loss, dpi=150)

    # === Post-warmup summary (optional): medians where lambda_warm ~ 1 ===
    try:
        post = df[df["lambda_warm"] >= 0.999]
        if len(post) > 0:
            med_reff = np.nanmedian(post["r_eff"].to_numpy())
            med_pred = np.nanmedian(post["cfd_fro_pred"].to_numpy())
            med_shrk = np.nanmedian(post["cfd_fro_shrunk"].to_numpy())
            med_smth = np.nanmedian(post["cfd_fro_smooth"].to_numpy())
            print(f"[post-warmup medians | n={len(post)}]")
            print(f"  r_eff:            {med_reff:8.4g}")
            print(f"  fro_pred:         {med_pred:8.4g}")
            print(f"  fro_shrunk:       {med_shrk:8.4g}")
            print(f"  fro_smooth:       {med_smth:8.4g}")
        else:
            print("[post-warmup medians] no rows with lambda_warm ≥ 0.999 yet.")
    except Exception as e:
        print("[post-warmup medians] skipped due to error:", e)



    print("Saved:")
    print(" ", out1)
    print(" ", out2)
    if out3: print(" ", out3)
    if out4: print(" ", out4)
    if out5: print(" ", out5)
    if out6: print(" ", out6)
    if out7: print(" ", out7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="training_log.csv", help="path to training_log.csv")
    parser.add_argument("--smooth", type=int, default=200, help="rolling window size (set 0/1 to disable)")
    parser.add_argument("--clip", type=float, default=0.0, help="clip |r_eff| at this value for plotting (0 = no clip)")
    parser.add_argument("--bins", type=int, default=100, help="histogram bins for r_eff")
    args = parser.parse_args()
    main(args.csv, args.smooth if args.smooth and args.smooth > 1 else None,
         args.clip if args.clip and args.clip > 0 else None,
         args.bins)

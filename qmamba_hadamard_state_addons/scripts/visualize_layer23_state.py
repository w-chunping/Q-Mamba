#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_layer(states_path, layer_idx=23):
    obj = torch.load(states_path, map_location="cpu")
    states = obj["states"]   # expected: list of [T, B, H, P, N]
    x = states[layer_idx].float()
    return x


def save_heatmap(mat, out_path, title, cbar_label):
    plt.figure(figsize=(7, 4.8))
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.colorbar(label=cbar_label)
    plt.xlabel("State dimension N")
    plt.ylabel("Channel / headdim P")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/layer23_viz")
    parser.add_argument("--layer_idx", type=int, default=23)

    # choose one snapshot/head for raw heatmap
    parser.add_argument("--snapshot_idx", type=int, default=0)
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--head_idx", type=int, default=0)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    x = load_layer(args.states, layer_idx=args.layer_idx)
    # expected shape [T, B, H, P, N]
    print("loaded layer shape:", tuple(x.shape))

    T, B, H, P, N = x.shape

    t = min(args.snapshot_idx, T - 1)
    b = min(args.batch_idx, B - 1)
    h = min(args.head_idx, H - 1)

    # --------------------------------------------------
    # 1. Raw heatmap: one snapshot / one head
    # --------------------------------------------------
    raw = x[t, b, h]   # [P, N]
    raw_logabs = torch.log10(raw.abs() + 1e-12).numpy()

    save_heatmap(
        raw_logabs,
        out_path=f"{args.out_dir}/layer{args.layer_idx}_raw_logabs_t{t}_h{h}.png",
        title=f"Layer {args.layer_idx} raw state log10(|h|), snapshot {t}, head {h}",
        cbar_label="log10(|state value|)",
    )

    # --------------------------------------------------
    # 2. Max magnitude aggregated over T and H
    #    shape: [P, N]
    # --------------------------------------------------
    max_abs = x[:, b].abs().amax(dim=(0, 1))  # over T,H -> [P,N]
    max_abs_log = torch.log10(max_abs + 1e-12).numpy()

    save_heatmap(
        max_abs_log,
        out_path=f"{args.out_dir}/layer{args.layer_idx}_maxabs_log.png",
        title=f"Layer {args.layer_idx} max over snapshots/heads: log10(max |h|)",
        cbar_label="log10(max |state value|)",
    )

    # --------------------------------------------------
    # 3. RMS aggregated over T and H
    # --------------------------------------------------
    rms = torch.sqrt(torch.mean(x[:, b] ** 2, dim=(0, 1)) + 1e-12)  # [P,N]
    rms_log = torch.log10(rms + 1e-12).numpy()

    save_heatmap(
        rms_log,
        out_path=f"{args.out_dir}/layer{args.layer_idx}_rms_log.png",
        title=f"Layer {args.layer_idx} RMS over snapshots/heads",
        cbar_label="log10(RMS)",
    )

    # --------------------------------------------------
    # 4. Outlier score map: max / RMS
    # --------------------------------------------------
    outlier_score = max_abs / (rms + 1e-12)
    outlier_score_log = torch.log10(outlier_score + 1e-12).numpy()

    save_heatmap(
        outlier_score_log,
        out_path=f"{args.out_dir}/layer{args.layer_idx}_outlier_score_log.png",
        title=f"Layer {args.layer_idx} outlier score log10(max/RMS)",
        cbar_label="log10(max/RMS)",
    )

    # --------------------------------------------------
    # 5. p99.9 / RMS map (more stable than max/RMS)
    # --------------------------------------------------
    # flatten over T and H for each (P,N)
    xb = x[:, b]  # [T,H,P,N]
    flat = xb.permute(2, 3, 0, 1).reshape(P, N, -1)   # [P,N,T*H]
    p999 = torch.quantile(flat.abs(), 0.999, dim=-1)  # [P,N]
    p999_score = p999 / (rms + 1e-12)
    p999_score_log = torch.log10(p999_score + 1e-12).numpy()

    save_heatmap(
        p999_score_log,
        out_path=f"{args.out_dir}/layer{args.layer_idx}_p999_over_rms_log.png",
        title=f"Layer {args.layer_idx} outlier score log10(p99.9/RMS)",
        cbar_label="log10(p99.9/RMS)",
    )

    print(f"saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
"""Phase 12 figures from tables/tables.json (p12_tables.py), matplotlib only.

    python -m vint_train.mapmad.closed_loop.p12_figures --config <p12_exam.yaml>      # -> <out_dir>/figures/*.png

1. success.png: oracle and rule-stop success per main-exam arm with the home-clustered 95% CI;
2. bins.png: oracle success per distance bin, per arm;
3. house_seen.png: exploration side test, house seen at 250 / 500 / 1,000 steps per arm (with CI);
4. tau.png: dev rule-stop success per tau (the tau rule; P12_DEV_CHOICES.json).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mapmad_sim.run_layout import RunLayout  # noqa: E402


def bar_ci(ax, labels, vals, color, offset, width, name):
    m = np.array([v[0] for v in vals], float)
    err = np.array([[v[0] - v[1], v[2] - v[0]] for v in vals], float).T
    x = np.arange(len(labels)) + offset
    ax.bar(x, m, width, yerr=err, capsize=3, color=color, label=name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    t = json.loads((layout.base / "tables" / "tables.json").read_text())
    out = layout.base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    arms = t["main"]["arms"]
    names = list(arms)
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_ci(ax, names, [arms[a]["oracle_success"] for a in names], "#4C72B0", -0.2, 0.4, "oracle stop")
    bar_ci(ax, names, [arms[a]["rule_success"] for a in names], "#DD8452", 0.2, 0.4, f"rule stop (tau {t['tau']})")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("success (episode mean of 3 seeds)")
    ax.set_title(f"Phase 12 main exam (headline {t['headline']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "success.png", dpi=120)
    plt.close(fig)

    bins = t["bins"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["0", "1", "2"]
    for k, a in enumerate(bins):
        ax.plot([0, 1, 2], [bins[a].get(b, {}).get("oracle") or 0 for b in labels], marker="o", label=a)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["3-6 m", "6-10 m", "10-15 m"])
    ax.set_ylabel("oracle success")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "bins.png", dpi=120)
    plt.close(fig)

    ex = {a: v for a, v in t.get("explore", {}).items() if "-" not in a}
    if ex:
        fig, ax = plt.subplots(figsize=(8, 4))
        for a, v in ex.items():
            m = [v[s][0] for s in ("250", "500", "1000")]
            lo = [v[s][1] for s in ("250", "500", "1000")]
            hi = [v[s][2] for s in ("250", "500", "1000")]
            ax.errorbar([250, 500, 1000], m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], marker="o", capsize=3, label=a)
        ax.set_xlabel("steps")
        ax.set_ylabel("house seen (share of walkable cells)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "house_seen.png", dpi=120)
        plt.close(fig)

    ch_f = layout.base / "P12_DEV_CHOICES.json"
    if ch_f.exists():
        ch = json.loads(ch_f.read_text())
        by = ch.get("rule_stop_success_by_tau") or {}
        if by:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot([float(k) for k in by], list(by.values()), marker="o")
            ax.axvline(float(ch["tau"]), color="r", ls="--", label=f"tau = {ch['tau']:.2f}")
            ax.set_xlabel("tau")
            ax.set_ylabel("rule-stop success (dev, headline + YOLOE)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "tau.png", dpi=120)
            plt.close(fig)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()

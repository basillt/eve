"""
analyze.py — does a PERSONAL hot core (experts or neurons) beat a GENERIC one?

Idea: keep only the "hot" experts of each layer in RAM, leave the rest on SSD.
A "hit" is an expert activation served from RAM. We compare two ways to pick
the hot experts, both evaluated on personal prompts the core has NOT seen:

  personal core : built from your own (train) prompts
  generic core  : built from general prompts (what existing systems do)
  oracle        : built from the test prompts themselves (upper bound)
  random        : no knowledge (lower bound = budget fraction)

Usage:
    python analyze.py --personal results/personal.npz --generic results/generic.npz
"""

import argparse
import json
import os

import numpy as np


def load(path):
    d = np.load(path, allow_pickle=False)
    return d["counts"].astype(np.int64), d["tokens"], json.loads(str(d["meta"]))


def hit_curve(build_counts, test_counts):
    """Hit rate for every budget k = 1..E experts per layer.

    build_counts, test_counts: [L, E] summed activation counts.
    Returns array of length E: fraction of test activations served by the
    top-k experts (ranked per layer by build_counts).
    """
    L, E = build_counts.shape
    # Small random tie-break so unseen experts are not always picked in index order.
    noise = np.random.default_rng(0).random((L, E)) * 1e-6
    order = np.argsort(-(build_counts + noise), axis=1)            # [L, E] hottest first
    served = np.take_along_axis(test_counts, order, axis=1)          # test counts in that order
    cum = np.cumsum(served, axis=1).sum(axis=0)                      # [E] summed over layers
    return cum / max(test_counts.sum(), 1)


def budget_for(curve, target):
    """Smallest number of experts per layer reaching the target hit rate."""
    idx = np.argmax(curve >= target)
    return int(idx + 1) if curve[idx] >= target else len(curve)


def ram_gb(meta, k, bits):
    """RAM for a core with k of E experts per layer, at a given bits/weight."""
    if "expert_params" not in meta:
        return None
    E = meta["num_experts"]
    params = meta["non_expert_params"] + meta["expert_params"] * k / E
    return params * bits / 8 / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--personal", required=True, help="npz from profile_experts.py on YOUR prompts")
    ap.add_argument("--generic", required=True, help="npz from profile_experts.py on general prompts")
    ap.add_argument("--train-frac", type=float, default=0.7, help="share of personal prompts used to build the core")
    ap.add_argument("--splits", type=int, default=20, help="random train/test splits to average over")
    ap.add_argument("--split", choices=["random", "time"], default="random",
                    help="time = build the core from the first prompts in the file, test on the later ones "
                         "(use with chronological prompts from import_chats.py — the honest test)")
    ap.add_argument("--targets", default="0.8,0.9,0.95,0.99")
    ap.add_argument("--cross", help="npz of ANOTHER person's prompts: checks that your core is specific to you")
    ap.add_argument("--bits", type=float, default=4.5, help="bits per weight for the RAM estimate (Q4 ≈ 4.5)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    P, _, meta_p = load(args.personal)
    G, _, meta_g = load(args.generic)
    if P.shape[1:] != G.shape[1:] or meta_p.get("unit", "experts") != meta_g.get("unit", "experts"):
        raise SystemExit("personal and generic files come from different models or units")
    unit = meta_p.get("unit", "experts")
    if bool(meta_p.get("skip_template")) != bool(meta_g.get("skip_template")):
        print("[warn] only one of the two files skipped template tokens — re-run both the same way")
    n, L, E = P.shape
    if n < 4:
        raise SystemExit("need at least 4 personal prompts")
    targets = [float(t) for t in args.targets.split(",")]

    generic_build = G.sum(axis=0)
    rng = np.random.default_rng(42)
    curves = {"personal": [], "generic": [], "oracle": []}
    C = load(args.cross)[0] if args.cross else None
    if C is not None and C.shape[1:] != P.shape[1:]:
        raise SystemExit("--cross file comes from a different model")
    cross_curves = []
    n_train = max(1, min(n - 1, int(round(n * args.train_frac))))
    for _ in range(1 if args.split == "time" else args.splits):
        perm = np.arange(n) if args.split == "time" else rng.permutation(n)
        train, test = P[perm[:n_train]].sum(axis=0), P[perm[n_train:]].sum(axis=0)
        curves["personal"].append(hit_curve(train, test))
        curves["generic"].append(hit_curve(generic_build, test))
        curves["oracle"].append(hit_curve(test, test))
        if C is not None:
            cross_curves.append(hit_curve(train, C.sum(axis=0)))
    mean = {k: np.mean(v, axis=0) for k, v in curves.items()}
    mean["random"] = np.arange(1, E + 1) / E

    # ---------------------------------------------------------------- report
    meta = meta_p
    full_gb = ram_gb(meta, E, args.bits)
    what = (f"{L} MoE layers × {E} experts, top-{meta['top_k']}" if unit == "experts" else
            f"{L} MLP layers × {E} neurons, top {meta.get('neuron_frac', 0):.0%} per token")
    lines = [f"model: {meta['model']}   ({what}){'   [template tokens skipped]' if meta.get('skip_template') else ''}",
             f"personal prompts: {n}  (train {args.train_frac:.0%}, "
             f"{'time split' if args.split == 'time' else f'{args.splits} random splits'})   "
             f"generic prompts: {G.shape[0]}",
             ""]
    header = f"{'target hit':>10} | {'personal core':>22} | {'generic core':>22} | {'oracle':>8} | saving vs generic"
    lines += [header, "-" * len(header)]
    summary = {"model": meta["model"], "unit": unit, "num_experts": E, "num_layers": L,
               "personal_prompts": n, "generic_prompts": int(G.shape[0]), "targets": {}}
    for t in targets:
        kp, kg, ko = (budget_for(mean[c], t) for c in ("personal", "generic", "oracle"))
        gp, gg = ram_gb(meta, kp, args.bits), ram_gb(meta, kg, args.bits)

        def cell(k, gb):
            s = f"{k}/{E} ({k / E:.0%})"
            return s + (f" {gb:.2f}GB" if gb is not None else "")

        saving = f"{kg / kp:.2f}× fewer {unit}" if kp else "-"
        lines.append(f"{t:>10.0%} | {cell(kp, gp):>22} | {cell(kg, gg):>22} | {ko / E:>8.0%} | {saving}")
        summary["targets"][str(t)] = {"personal_k": kp, "generic_k": kg, "oracle_k": ko,
                                      "personal_ram_gb": gp, "generic_ram_gb": gg}
    if full_gb is not None:
        lines += ["", f"full model at {args.bits} bits/weight: {full_gb:.2f} GB "
                      f"(part that is always in RAM: {ram_gb(meta, 0, args.bits):.2f} GB)"]

    k20 = max(1, round(E * 0.2))
    gap = mean["personal"][k20 - 1] - mean["generic"][k20 - 1]
    lines += ["", f"with 20% of {unit} in RAM: personal core hits {mean['personal'][k20 - 1]:.1%}, "
                  f"generic core hits {mean['generic'][k20 - 1]:.1%}  (gap {gap * 100:+.1f} points)"]
    verdict = ("PROMISING: the personal core clearly beats the generic one."
               if gap >= 0.05 else
               "WEAK: personal and generic cores are close — personalization adds little here.")
    lines += [f"verdict: {verdict}"]
    k_or = budget_for(mean["oracle"], 0.9)
    lines += [f"best possible (oracle) for 90%: {k_or}/{E} {unit} ({k_or / E:.0%}) — "
              + ("the model itself is concentrated: a small core can work" if k_or / E <= 0.4 else
                 "the model spreads its work evenly: no small core can reach 90%")]
    summary["gap_at_20pct"] = float(gap)
    if C is not None:
        cross = np.mean(cross_curves, axis=0)
        cg = hit_curve(generic_build, C.sum(axis=0))
        lines += ["", f"cross-check with another person's prompts ({C.shape[0]}), 20% of {unit} in RAM:",
                  f"  YOUR core on YOUR new prompts:        {mean['personal'][k20 - 1]:.1%}",
                  f"  YOUR core on THE OTHER person:        {cross[k20 - 1]:.1%}",
                  f"  generic core on THE OTHER person:     {cg[k20 - 1]:.1%}",
                  "  -> " + ("your core is specific to you (it serves you much better than someone else)"
                             if mean['personal'][k20 - 1] - cross[k20 - 1] >= 0.05 else
                             "your core is not very specific to you")]
        summary["cross"] = {"own": float(mean['personal'][k20 - 1]), "other": float(cross[k20 - 1]),
                            "generic_on_other": float(cg[k20 - 1])}
    summary["curves"] = {k: v.tolist() for k, v in mean.items()}

    os.makedirs(args.outdir, exist_ok=True)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot(mean, E, meta, os.path.join(args.outdir, "hit_rate.png"))


def plot(mean, E, meta, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed — skipping chart")
        return

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    x = np.arange(1, E + 1) / E * 100
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    # References first, personal last so it sits on top.
    series = [("oracle", "Oracle (upper bound)", ink2, "--"),
              ("random", "Random (lower bound)", ink2, ":"),
              ("generic", "Generic core (general prompts)", "#eb6834", "-"),
              ("personal", "Personal core (your prompts)", "#2a78d6", "-")]
    for key, label, color, style in series:
        y = mean[key] * 100
        ax.plot(x, y, style, color=color, lw=2, label=label)
        if key in ("personal", "generic"):
            ax.annotate(label.split(" (")[0], (x[len(x) // 4], y[len(y) // 4]),
                        xytext=(6, -14 if key == "generic" else 6), textcoords="offset points",
                        color=ink, fontsize=9)

    ax.axhline(90, color=grid, lw=1, zorder=0)
    ax.text(1, 90.5, "90% hit rate", ha="left", va="bottom", color=ink2, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.set_xlabel(f"{meta.get('unit', 'experts').capitalize()} kept in RAM per layer (%)", color=ink2)
    ax.set_ylabel("Expert activations served from RAM (%)", color=ink2)
    ax.set_title(f"Personal vs generic hot {meta.get('unit', 'experts')[:-1]} core — {meta['model'].split('/')[-1]}",
                 color=ink, fontsize=11, loc="left")
    ax.grid(True, color=grid, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=ink2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc="lower right", fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    print(f"[chart] {path}")


if __name__ == "__main__":
    main()

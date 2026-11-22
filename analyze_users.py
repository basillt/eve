"""
analyze_users.py — personal vs population expert cores across MANY real users.

For every user (prompts in time order):
  personal core    built from the user's own first 70% of prompts
  population core  built from all OTHER users' first 70% (each user weighted equally)
  oracle           built from the user's own last 30% (upper bound)
All three are evaluated on the user's last 30% — prompts the cores never saw.

Cache simulation (needs --trace from profile_experts.py --trace-out): replays the
user's later prompts token by token through a per-layer expert cache of a fixed
size and compares:
  static population   fixed generic core
  static personal     fixed personal core
  LRU                 dynamic cache that follows the current request (stand-in for
                      request-level caching such as MoE-Infinity; NOT their full system)
  LRU + personal      same LRU, but warm-started from the personal core
  pinned + LRU        half the cache pinned to the personal core, half LRU

    python analyze_users.py --counts results_users/all.npz --prompts prompts/wildchat_users.jsonl \\
        --trace results_users/trace.npz --outdir results_users
"""

import argparse
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np

from analyze import hit_curve, budget_for

BUDGETS = [0.1, 0.2, 0.3, 0.5]


def load_users(path):
    users = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if (obj.get("text") or obj.get("prompt") or "").strip():
                users.append(obj.get("user", "u?"))
    return users


def top_sets(build, k):
    noise = np.random.default_rng(1).random(build.shape) * 1e-9
    order = np.argsort(-(build + noise), axis=1)
    return order


def simulate(trace, offsets, test_idx, order_static, cap, policy, order_personal=None):
    """Hit rate of expert accesses on the user's test prompts, per-layer cache of `cap` experts."""
    L = trace.shape[1]
    hits = total = 0
    caches, pinned = [], []
    for layer in range(L):
        if policy in ("static_pop", "static_personal"):
            caches.append(set(order_static[layer, :cap].tolist()))
            pinned.append(None)
        elif policy == "lru_pop":
            caches.append(OrderedDict((int(e), None) for e in order_static[layer, :cap]))
            pinned.append(set())
        elif policy == "lru_personal":
            caches.append(OrderedDict((int(e), None) for e in order_personal[layer, :cap]))
            pinned.append(set())
        elif policy == "pinned_lru":
            p = cap // 2
            pinned.append(set(order_personal[layer, :p].tolist()))
            caches.append(OrderedDict((int(e), None) for e in order_personal[layer, p:cap]))
        else:
            raise ValueError(policy)
    lru_cap = [cap - (len(pinned[l]) if pinned[l] else 0) for l in range(L)]

    for i in test_idx:
        seg = trace[offsets[i]:offsets[i + 1]]          # [tokens, L, K]
        for tok in seg:
            for layer in range(L):
                cache = caches[layer]
                for e in tok[layer]:
                    e = int(e)
                    total += 1
                    if pinned[layer] is None:                # static
                        hits += e in cache
                        continue
                    if e in pinned[layer]:
                        hits += 1
                    elif e in cache:
                        hits += 1
                        cache.move_to_end(e)
                    else:
                        if lru_cap[layer] > 0:
                            cache[e] = None
                            if len(cache) > lru_cap[layer]:
                                cache.popitem(last=False)
    return hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--trace", default="")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cache-budgets", default="0.2,0.35")
    ap.add_argument("--outdir", default="results_users")
    args = ap.parse_args()

    d = np.load(args.counts)
    counts = d["counts"].astype(np.float64)
    meta = json.loads(str(d["meta"]))
    users = load_users(args.prompts)
    if len(users) != counts.shape[0]:
        raise SystemExit(f"prompts ({len(users)}) and counts ({counts.shape[0]}) do not line up")
    P, L, E = counts.shape
    by_user = defaultdict(list)
    for i, u in enumerate(users):
        by_user[u].append(i)
    names = sorted(by_user)
    split = {}
    for u in names:
        idx = by_user[u]
        n_train = max(1, min(len(idx) - 1, int(round(len(idx) * args.train_frac))))
        split[u] = (idx[:n_train], idx[n_train:])

    train_sum = {u: counts[split[u][0]].sum(axis=0) for u in names}
    test_sum = {u: counts[split[u][1]].sum(axis=0) for u in names}
    train_norm = {u: train_sum[u] / max(train_sum[u].sum(), 1) for u in names}
    all_norm = sum(train_norm.values())

    rows, cross = [], []
    curves = {"personal": [], "population": [], "oracle": []}
    for u in names:
        pop = all_norm - train_norm[u]                  # leave-one-out population core
        cp = hit_curve(train_sum[u], test_sum[u])
        cg = hit_curve(pop, test_sum[u])
        co = hit_curve(test_sum[u], test_sum[u])
        curves["personal"].append(cp); curves["population"].append(cg); curves["oracle"].append(co)
        k20 = max(1, round(E * 0.2))
        others = [hit_curve(train_sum[u], test_sum[v])[k20 - 1] for v in names if v != u]
        cross.append(float(np.mean(others)))
        rows.append({"user": u, "n_prompts": len(by_user[u]),
                     **{f"personal@{b}": float(cp[max(1, round(E * b)) - 1]) for b in BUDGETS},
                     **{f"population@{b}": float(cg[max(1, round(E * b)) - 1]) for b in BUDGETS},
                     **{f"oracle@{b}": float(co[max(1, round(E * b)) - 1]) for b in BUDGETS},
                     "k90_personal": budget_for(cp, 0.9), "k90_population": budget_for(cg, 0.9),
                     "k90_oracle": budget_for(co, 0.9), "cross@0.2": cross[-1]})

    mean = {k: np.mean(v, axis=0) for k, v in curves.items()}
    n = len(names)

    def ms(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

    lines = [f"model: {meta['model']}  ({L} MoE layers x {E} experts, top-{meta['top_k']})",
             f"users: {n}   prompts: {P}   (per user: first {args.train_frac:.0%} builds the core, rest tests it)",
             "", "share of expert activations served from the core (mean over users ± standard error)",
             f"{'budget':>8} | {'personal':>15} | {'population':>15} | {'oracle':>8} | random | users where personal wins"]
    lines.append("-" * len(lines[-1]))
    for b in BUDGETS:
        mp, sp = ms(f"personal@{b}")
        mg, sg = ms(f"population@{b}")
        mo, _ = ms(f"oracle@{b}")
        wins = sum(r[f"personal@{b}"] > r[f"population@{b}"] for r in rows)
        lines.append(f"{b:>8.0%} | {mp:>8.1%} ±{sp:5.1%} | {mg:>8.1%} ±{sg:5.1%} | {mo:>8.1%} | {b:>6.0%} | {wins}/{n}")
    kp, kg, ko = (np.mean([r[k] for r in rows]) for k in ("k90_personal", "k90_population", "k90_oracle"))
    lines += ["", f"experts per layer needed for 90%: personal {kp:.1f}/{E}, population {kg:.1f}/{E}, oracle {ko:.1f}/{E}",
              f"cross-check @20%: a user's core on OTHER users' prompts {np.mean(cross):.1%} "
              f"(on the user's own later prompts {ms('personal@0.2')[0]:.1%})"]

    summary = {"model": meta["model"], "users": n, "prompts": P, "rows": rows,
               "curves": {k: v.tolist() for k, v in mean.items()}, "cross@0.2": float(np.mean(cross))}

    if args.trace:
        t = np.load(args.trace)
        trace, offsets = t["trace"], t["offsets"]
        if len(offsets) - 1 != P:
            raise SystemExit("trace does not line up with the prompts")
        policies = [("static_pop", "static population core"), ("static_personal", "static personal core"),
                    ("lru_pop", "LRU (request-level, generic warm start)"),
                    ("lru_personal", "LRU warm-started from personal core"),
                    ("pinned_lru", "half pinned personal + half LRU")]
        cache_res = {}
        lines += ["", "cache simulation on each user's later prompts (hit rate of expert accesses, mean over users)"]
        for b in [float(x) for x in args.cache_budgets.split(",")]:
            cap = max(1, round(E * b))
            res = {p: [] for p, _ in policies}
            for u in names:
                pop = all_norm - train_norm[u]
                o_pop, o_per = top_sets(pop, cap), top_sets(train_sum[u], cap)
                for p, _ in policies:
                    static = o_per if p == "static_personal" else o_pop
                    res[p].append(simulate(trace, offsets, split[u][1], static, cap, p, o_per))
            lines.append(f"  cache = {cap}/{E} experts per layer ({b:.0%}):")
            for p, label in policies:
                v = np.array(res[p])
                lines.append(f"    {label:<42} {v.mean():6.1%}  (±{v.std(ddof=1) / np.sqrt(len(v)):.1%})")
            best_dyn = np.array(res["lru_pop"])
            for p in ("lru_personal", "pinned_lru"):
                gain = np.array(res[p]) - best_dyn
                lines.append(f"    -> {p} vs LRU: {gain.mean() * 100:+.1f} points, better for "
                             f"{int((gain > 0).sum())}/{n} users")
            cache_res[str(b)] = {p: float(np.mean(v)) for p, v in res.items()}
        summary["cache"] = cache_res

    os.makedirs(args.outdir, exist_ok=True)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.outdir, "users_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(args.outdir, "users_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    plot(mean, E, meta, n, os.path.join(args.outdir, "users_hit_rate.png"))


def plot(mean, E, meta, n, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    x = np.arange(1, E + 1) / E * 100
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
    ax.plot(x, mean["oracle"] * 100, "--", color=ink2, lw=2, label="Oracle (upper bound)")
    ax.plot(x, x, ":", color=ink2, lw=2, label="Random (lower bound)")
    ax.plot(x, mean["population"] * 100, "-", color="#eb6834", lw=2, label="Population core (other users)")
    ax.plot(x, mean["personal"] * 100, "-", color="#2a78d6", lw=2, label="Personal core (same user, earlier prompts)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    ax.set_xlabel("Experts kept in RAM per layer (%)", color=ink2)
    ax.set_ylabel("Expert activations served from RAM (%)", color=ink2)
    ax.set_title(f"{n} real users (WildChat) — {meta['model'].split('/')[-1]}", color=ink, fontsize=11, loc="left")
    ax.grid(True, color=grid, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=ink2)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, loc="lower right", fontsize=9, labelcolor=ink)
    fig.tight_layout(); fig.savefig(path, facecolor=surface)
    print(f"[chart] {path}")


if __name__ == "__main__":
    main()

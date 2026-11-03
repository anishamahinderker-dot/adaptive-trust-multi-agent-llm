# =====================================================================
# PHASE 6: make_plots.py
# Makes the figures for the paper + a statistical significance test.
# Uses only saved CSV files (no models, no GPU) -> takes seconds.
#
# Two sets of results are used:
#   FULL set  (all answered questions, e.g. 1319): robustness, trust
#             evolution and significance tests
#   200 set   (first 200 questions): comparison with DEBATE, because
#             debate was only run on those 200
#
# Before running, create the result files:
#   python compute_results.py
#   python compute_results.py bad
#   python compute_results.py 200
#   python compute_results.py bad 200
# Then:
#   python make_plots.py
#
# Output: "figures" folder (300 dpi PNGs) + significance.csv
# =====================================================================

import os
import math
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3:mini"]
BAD = "qwen2.5:0.5b"
SEED = 42
SHORT = {"qwen2.5:3b": "Qwen2.5-3B", "llama3.2:3b": "Llama3.2-3B",
         "gemma2:2b": "Gemma2-2B", "phi3:mini": "Phi-3-mini", "qwen2.5:0.5b": "Qwen2.5-0.5B (bad)"}

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

# FULL-set files
res_main = pd.read_csv("results_main.csv")
res_bad = pd.read_csv("results_with_bad_agent.csv")
pq_main = pd.read_csv("per_question_main.csv")
pq_bad = pd.read_csv("per_question_with_bad_agent.csv")
N_FULL = len(pq_main)

# 200-set files (fall back to full set if they don't exist)
if os.path.exists("results_main_200.csv"):
    res_200 = pd.read_csv("results_main_200.csv")
    pq_200 = pd.read_csv("per_question_main_200.csv")
else:
    res_200, pq_200 = res_main, pq_main
N_200 = len(pq_200)
print(f"Full set: {N_FULL} questions | Debate comparison set: {N_200} questions\n")


def get(table, starts_with):
    row = table[table["Method"].str.startswith(starts_with)].iloc[0]
    return row["Accuracy"], row["Avg tokens per question"]


# ------------------------- DEBATE (if finished) ----------------------
def debate_per_question():
    if not os.path.exists("debate_answers.csv"):
        return None, None, None
    d = pd.read_csv("debate_answers.csv")
    r1 = pd.read_csv("answers.csv")
    dmap = {(r["agent"], r["q_id"]): r for _, r in d.iterrows()}
    r1tok = {(r["agent"], r["q_id"]): r["tokens"] for _, r in r1.iterrows()}

    order = list(range(1319))
    random.Random(SEED).shuffle(order)
    order = [q for q in order[:200] if all((a, q) in dmap for a in MAIN)]
    if len(order) < 200:
        print(f"Debate not finished yet ({len(order)}/200 questions). Skipping debate.\n")
        return None, None, None

    rng = random.Random(SEED)
    correct, tok = {}, 0
    for q in order:
        votes = {}
        for a in MAIN:
            p = dmap[(a, q)]["round2_answer"]
            if not pd.isna(p):
                p = round(float(p), 4)
                votes[p] = votes.get(p, 0) + 1
        final = None
        if votes:
            top = max(votes.values())
            final = rng.choice(sorted(p for p, v in votes.items() if v == top))
        correct[q] = int(final == round(float(dmap[(MAIN[0], q)]["correct_answer"]), 4))
        tok += sum(r1tok[(a, q)] + dmap[(a, q)]["tokens"] for a in MAIN)
    return correct, round(100 * sum(correct.values()) / len(order), 1), round(tok / len(order))


deb_q, deb_acc, deb_tok = debate_per_question()

# ------------------------- FIGURE A: accuracy (200 set, with debate) -
labels = [SHORT[a] for a in MAIN] + ["Majority\nvote", "Trust\n(consensus)", "Trust\n(verified)"]
values = [get(res_200, "Single: " + a)[0] for a in MAIN] + [
    get(res_200, "Majority")[0], get(res_200, "Trust-weighted (consensus")[0],
    get(res_200, "Trust-weighted (oracle")[0]]
colors = ["#9e9e9e"] * 4 + ["#5b8db8", "#2e7d32", "#81c784"]
if deb_acc is not None:
    labels.append("Debate\n(2 rounds)")
    values.append(deb_acc)
    colors.append("#e08e45")

fig, ax = plt.subplots(figsize=(8.5, 4))
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("Accuracy on GSM8K (%)")
ax.set_ylim(0, 100)
ax.set_title(f"Accuracy of single agents and multi-agent methods ({N_200} GSM8K problems)")
plt.xticks(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig_accuracy.png", dpi=300)
plt.close()

# ------------------------- FIGURE B: robustness (full set) -----------
methods = ["Majority vote", "Trust (consensus)", "Trust (verified)"]
keys = ["Majority", "Trust-weighted (consensus", "Trust-weighted (oracle"]
without = [get(res_main, k)[0] for k in keys]
with_bad = [get(res_bad, k)[0] for k in keys]

fig, ax = plt.subplots(figsize=(6, 4))
x = range(len(methods))
w = 0.35
b1 = ax.bar([i - w / 2 for i in x], without, w, label="4 agents", color="#5b8db8", edgecolor="black", linewidth=0.5)
b2 = ax.bar([i + w / 2 for i in x], with_bad, w, label="4 agents + 1 unreliable agent", color="#e57373", edgecolor="black", linewidth=0.5)
for bars_ in (b1, b2):
    for b in bars_:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.1f}", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(methods)
ax.set_ylabel("Accuracy (%)")
low = min(without + with_bad)
ax.set_ylim(max(0, low - 6), min(100, max(without + with_bad) + 3))  # zoomed; axis labelled
ax.set_title(f"Effect of adding an unreliable agent ({N_FULL} problems)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig_robustness.png", dpi=300)
plt.close()

# ------------------------- FIGURE C: trust over time (full set) ------
# Raw trust jumps a lot (lambda = 0.9 remembers ~10 problems), so we plot a
# 50-problem moving average to make the trend readable. Say so in the caption.
SMOOTH = 50
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, mode, title in zip(axes, ["consensus", "oracle"],
                           ["Consensus feedback (no labels)", "Verification feedback"]):
    h = pd.read_csv(f"trust_history_with_bad_agent_{mode}.csv")
    for a in MAIN + [BAD]:
        smooth = h[a].rolling(SMOOTH, min_periods=1).mean()
        ax.plot(h["step"], smooth, "--" if a == BAD else "-", label=SHORT[a], linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Problems processed")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel(f"Trust score ({SMOOTH}-problem moving average)")
handles, labs = axes[0].get_legend_handles_labels()
fig.legend(handles, labs, loc="upper center", ncol=5, fontsize=8, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.92])   # leave room for the legend on top
plt.savefig("figures/fig_trust_evolution.png", dpi=300)
plt.close()

# ------------------------- FIGURE D: accuracy vs cost (200 set) ------
pts = [(SHORT[a], *get(res_200, "Single: " + a)) for a in MAIN]
pts += [("Majority", *get(res_200, "Majority")),
        ("Trust (ours)", *get(res_200, "Trust-weighted (consensus"))]
if deb_acc is not None:
    pts.append(("Debate", deb_acc, deb_tok))

# Label positions chosen so close points don't overlap
OFFSET = {"Majority": (8, 5), "Trust (ours)": (8, -12), "Llama3.2-3B": (8, 6),
          "Qwen2.5-3B": (8, -12), "Debate": (-10, -14)}
fig, ax = plt.subplots(figsize=(6, 4))
for name, acc, tok in pts:
    ax.scatter(tok, acc, s=60, edgecolor="black")
    dx, dy = OFFSET.get(name, (8, 4))
    ax.annotate(name, (tok, acc), textcoords="offset points", xytext=(dx, dy), fontsize=8,
                ha="right" if dx < 0 else "left")
ax.set_xlabel("Average tokens per problem")
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Accuracy vs. computational cost ({N_200} problems)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig_cost.png", dpi=300)
plt.close()


# ------------------------- SIGNIFICANCE TEST (McNemar) ---------------
def mcnemar(a, b):
    only_a = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    only_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    return only_a, only_b, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


comparisons = [
    (f"4 agents (n={N_FULL})", "Trust (consensus)", "Majority vote", pq_main["trust_consensus"], pq_main["majority"]),
    (f"4 agents (n={N_FULL})", "Majority vote", "Best single (Phi-3-mini)", pq_main["majority"], pq_main["phi3:mini"]),
    (f"4 agents (n={N_FULL})", "Trust (consensus)", "Best single (Phi-3-mini)", pq_main["trust_consensus"], pq_main["phi3:mini"]),
    (f"+1 bad agent (n={N_FULL})", "Trust (consensus)", "Majority vote", pq_bad["trust_consensus"], pq_bad["majority"]),
    (f"+1 bad agent (n={N_FULL})", "Trust (verified)", "Majority vote", pq_bad["trust_oracle"], pq_bad["majority"]),
]
if deb_q is not None:
    deb_list = [deb_q[q] for q in pq_200["q_id"]]
    comparisons.append((f"4 agents (n={N_200})", "Debate", "Trust (consensus)", deb_list, pq_200["trust_consensus"]))
    comparisons.append((f"4 agents (n={N_200})", "Debate", "Majority vote", deb_list, pq_200["majority"]))

rows = []
for setting, m1, m2, a, b in comparisons:
    only1, only2, p = mcnemar(list(a), list(b))
    rows.append([setting, m1, m2, only1, only2, round(p, 4), "yes" if p < 0.05 else "no"])
sig = pd.DataFrame(rows, columns=["Setting", "Method A", "Method B", "Only A right",
                                  "Only B right", "p-value", "Significant (p<0.05)?"])
sig.to_csv("significance.csv", index=False)

print("===== SIGNIFICANCE TEST (McNemar, exact) =====")
print(sig.to_string(index=False))
if deb_acc is not None:
    print(f"\nDebate ({N_200} problems): accuracy {deb_acc}%, avg tokens {deb_tok}")
print("\nFigures saved in the 'figures' folder:")
for f in sorted(os.listdir("figures")):
    print("  ", f)

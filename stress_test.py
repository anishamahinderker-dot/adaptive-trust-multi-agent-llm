# =====================================================================
# STRESS TEST - step 2: stress_test.py
# 4 good agents + k unreliable agents, for k = 0, 1, 2, 3, 4.
# For each k: majority vote vs trust-weighted (consensus + oracle),
# plus McNemar significance test (trust vs majority).
#
# DESIGN FIXED IN ADVANCE: report these results whatever they show.
#
# Run:  python stress_test.py      (seconds, no GPU)
# Output: stress_results.csv + figures/fig_stress.png
# =====================================================================

import os
import sys
import math
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3:mini"]
WEAK = ["qwen2.5:0.5b", "qwen2.5:0.5b#s1", "qwen2.5:0.5b#s2", "qwen2.5:0.5b#s3"]
INIT_TRUST = 0.5
LAMBDA = 0.9      # same settings as the main experiment (not tuned)
SEED = 42

# ------------------------- LOAD ANSWERS ------------------------------
df = pd.read_csv("answers.csv")
df = df[df["agent"].isin(MAIN + WEAK)]
answer, truth = {}, {}
for _, r in df.iterrows():
    p = None if pd.isna(r["predicted_answer"]) else round(float(r["predicted_answer"]), 4)
    answer[(r["agent"], r["q_id"])] = p
    truth[r["q_id"]] = round(float(r["correct_answer"]), 4)

# Optional number = use only the first N questions:  python stress_test.py 200
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
SUFFIX = f"_{LIMIT}" if LIMIT else ""

order = list(range(1319))
random.Random(SEED).shuffle(order)
if LIMIT:
    order = order[:LIMIT]
order = [q for q in order if all((a, q) in answer for a in MAIN + WEAK)]
print(f"Using {len(order)} questions answered by all {len(MAIN + WEAK)} agents.\n")

print("Individual accuracy of the unreliable agents:")
for a in WEAK:
    acc = 100 * sum(answer[(a, q)] == truth[q] for q in order) / len(order)
    print(f"  {a:18s} {acc:.1f}%")
print()


# ------------------------- METHODS (same as compute_results.py) ------
def majority(agents):
    rng = random.Random(SEED)
    res = []
    for q in order:
        votes = {}
        for a in agents:
            p = answer[(a, q)]
            if p is not None:
                votes[p] = votes.get(p, 0) + 1
        final = None
        if votes:
            top = max(votes.values())
            final = rng.choice(sorted(p for p, v in votes.items() if v == top))
        res.append(int(final == truth[q]))
    return res


def trust_weighted(agents, mode):
    trust = {a: INIT_TRUST for a in agents}
    res = []
    for q in order:
        score = {}
        for a in agents:
            p = answer[(a, q)]
            if p is not None:
                score[p] = score.get(p, 0) + trust[a]
        final = max(score, key=score.get) if score else None
        res.append(int(final == truth[q]))
        for a in agents:
            p = answer[(a, q)]
            c = (1 if p == truth[q] else 0) if mode == "oracle" else (1 if (p is not None and p == final) else 0)
            trust[a] = LAMBDA * trust[a] + (1 - LAMBDA) * c
    return res


def mcnemar(a, b):
    only_a = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    only_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    return only_a, only_b, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ------------------------- RUN ALL k ---------------------------------
rows = []
for k in range(len(WEAK) + 1):
    agents = MAIN + WEAK[:k]
    maj = majority(agents)
    cons = trust_weighted(agents, "consensus")
    orac = trust_weighted(agents, "oracle")
    a1, b1, p_cons = mcnemar(cons, maj)
    a2, b2, p_orac = mcnemar(orac, maj)
    rows.append([k, round(100 * sum(maj) / len(order), 1),
                 round(100 * sum(cons) / len(order), 1), round(100 * sum(orac) / len(order), 1),
                 f"{a1} vs {b1}", round(p_cons, 4), f"{a2} vs {b2}", round(p_orac, 4)])

table = pd.DataFrame(rows, columns=["Unreliable agents (k)", "Majority %", "Trust (ours) %",
                                    "Trust (oracle) %", "Ours-only vs Maj-only right",
                                    "p (ours vs maj)", "Oracle-only vs Maj-only right",
                                    "p (oracle vs maj)"])
table.to_csv(f"stress_results{SUFFIX}.csv", index=False)
print("===== STRESS TEST RESULTS =====")
print(table.to_string(index=False))

# ------------------------- FIGURE ------------------------------------
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(table.iloc[:, 0], table["Majority %"], "o-", label="Majority vote", color="#5b8db8")
ax.plot(table.iloc[:, 0], table["Trust (ours) %"], "s-", label="Trust-weighted (consensus feedback)", color="#2e7d32")
ax.plot(table.iloc[:, 0], table["Trust (oracle) %"], "^--", label="Trust-weighted (verification feedback)", color="#81c784")
ax.set_xlabel("Number of unreliable agents added to 4 reliable agents")
ax.set_ylabel("Accuracy on GSM8K (%)")
ax.set_xticks(table.iloc[:, 0])
ax.set_title(f"Robustness as unreliable agents increase ({len(order)} problems)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"figures/fig_stress{SUFFIX}.png", dpi=300)
print(f"\nSaved: stress_results{SUFFIX}.csv, figures/fig_stress{SUFFIX}.png")

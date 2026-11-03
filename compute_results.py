# =====================================================================
# PHASE 3 (and Phase 5): compute_results.py
# Reads answers.csv (from Phase 2) and computes:
#   1. Single-agent accuracy (each model alone)
#   2. Majority vote (every agent's answer counts equally, NO trust)
#   3. Trust-weighted vote (OUR FRAMEWORK) with two feedback types:
#        - "consensus": trust goes up if the agent agreed with the
#          final answer. Needs NO correct answers -> realistic setting.
#        - "oracle": trust goes up if the agent was actually correct.
#          Uses the true answer -> best possible case ("upper bound").
#
# No models are run here, so it finishes in a few seconds.
#
# How to run:
#   python compute_results.py           -> the 4 main agents, all answered questions
#   python compute_results.py bad       -> 4 main agents + the bad agent
#   python compute_results.py 200       -> only the first 200 questions
#   python compute_results.py bad 200   -> bad agent, first 200 questions
#   (the "200" versions are used to compare with debate, which has 200 questions)
# =====================================================================

import sys
import random
import pandas as pd

# ------------------------- SETTINGS ----------------------------------
MAIN_AGENTS = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3:mini"]
BAD_AGENT = "qwen2.5:0.5b"

INIT_TRUST = 0.5   # every agent starts with the same trust (no favourites)
LAMBDA = 0.9       # memory of the trust update (see formula below)
SEED = 42          # same seed as Phase 2 -> same question order

# THE TRUST UPDATE RULE (this is the formula that goes in your paper):
#     T_i(new) = LAMBDA * T_i(old) + (1 - LAMBDA) * c_i
# c_i = 1 if the agent did well on this question, 0 if not.
# LAMBDA = 0.9 means: 90% old trust + 10% latest performance.
# So trust changes slowly and one lucky/unlucky answer doesn't decide it.

args = sys.argv[1:]
include_bad = "bad" in args
# Optional number = use only the first N questions (e.g. 200 to compare with debate)
LIMIT = next((int(a) for a in args if a.isdigit()), None)
AGENTS = MAIN_AGENTS + ([BAD_AGENT] if include_bad else [])
TAG = ("with_bad_agent" if include_bad else "main") + (f"_{LIMIT}" if LIMIT else "")

# ------------------------- LOAD ANSWERS ------------------------------
df = pd.read_csv("answers.csv")
df = df[df["agent"].isin(AGENTS)]

# Rebuild the exact same question order used in Phase 2
# (GSM8K test set has 1319 questions; same seed -> same shuffle)
order = list(range(1319))
random.Random(SEED).shuffle(order)
if LIMIT:
    order = order[:LIMIT]

# Only use questions that EVERY agent has answered
agents_per_q = df.groupby("q_id")["agent"].nunique()
complete = set(agents_per_q[agents_per_q == len(AGENTS)].index)
order = [q for q in order if q in complete]
print(f"Using {len(order)} questions answered by all {len(AGENTS)} agents.\n")
if len(order) == 0:
    sys.exit("No complete questions yet. Wait for Phase 2 to finish.")

# Put answers in simple lookup tables: (agent, question) -> value
answer, tokens, truth = {}, {}, {}
for _, r in df.iterrows():
    p = None if pd.isna(r["predicted_answer"]) else round(float(r["predicted_answer"]), 4)
    answer[(r["agent"], r["q_id"])] = p
    tokens[(r["agent"], r["q_id"])] = r["tokens"]
    truth[r["q_id"]] = round(float(r["correct_answer"]), 4)

# Tokens used per question when ALL agents answer (same for both voting methods)
all_agent_tokens = sum(tokens[(a, q)] for a in AGENTS for q in order) / len(order)

# ------------------------- 1. SINGLE AGENTS --------------------------
results = []
for a in AGENTS:
    acc = sum(answer[(a, q)] == truth[q] for q in order) / len(order)
    avg_tok = sum(tokens[(a, q)] for q in order) / len(order)
    results.append(["Single: " + a, acc, avg_tok])

# ------------------------- 2. MAJORITY VOTE --------------------------
rng = random.Random(SEED)  # used only to break ties fairly (randomly)


def majority_vote(q):
    votes = {}
    for a in AGENTS:
        p = answer[(a, q)]
        if p is not None:
            votes[p] = votes.get(p, 0) + 1
    if not votes:
        return None
    top = max(votes.values())
    tied = sorted(p for p, v in votes.items() if v == top)
    return rng.choice(tied)


maj_correct = {q: int(majority_vote(q) == truth[q]) for q in order}
maj_acc = sum(maj_correct.values()) / len(order)
results.append(["Majority vote (no trust)", maj_acc, all_agent_tokens])

# ------------------------- 3. TRUST-WEIGHTED VOTE --------------------
def trust_weighted(mode):
    """Go through the questions IN ORDER, like a real system would.
    For each question: vote using current trust -> then update trust."""
    trust = {a: INIT_TRUST for a in AGENTS}
    right = 0
    history = []
    per_q = {}   # was the final answer right on each question? (for significance test)

    for step, q in enumerate(order, start=1):
        # (a) Weighted vote: each answer gets the SUM of trust of agents giving it
        score = {}
        for a in AGENTS:
            p = answer[(a, q)]
            if p is not None:
                score[p] = score.get(p, 0) + trust[a]
        final = max(score, key=score.get) if score else None
        right += int(final == truth[q])
        per_q[q] = int(final == truth[q])

        # (b) Feedback: update every agent's trust
        for a in AGENTS:
            p = answer[(a, q)]
            if mode == "oracle":
                c = 1 if p == truth[q] else 0                      # was it correct?
            else:
                c = 1 if (p is not None and p == final) else 0     # agreed with final?
            trust[a] = LAMBDA * trust[a] + (1 - LAMBDA) * c

        row = {"step": step, "q_id": q}
        row.update(trust)
        history.append(row)

    # Save trust over time -> used for the trust graph in Phase 6
    pd.DataFrame(history).to_csv(f"trust_history_{TAG}_{mode}.csv", index=False)
    return right / len(order), trust, per_q


cons_acc, cons_trust, cons_q = trust_weighted("consensus")
orac_acc, orac_trust, orac_q = trust_weighted("oracle")

# Save right/wrong for every question and every method (used by make_plots.py)
per_question = pd.DataFrame({"q_id": order})
for a in AGENTS:
    per_question[a] = [int(answer[(a, q)] == truth[q]) for q in order]
per_question["majority"] = [maj_correct[q] for q in order]
per_question["trust_consensus"] = [cons_q[q] for q in order]
per_question["trust_oracle"] = [orac_q[q] for q in order]
per_question.to_csv(f"per_question_{TAG}.csv", index=False)
results.append(["Trust-weighted (consensus feedback) - OURS", cons_acc, all_agent_tokens])
results.append(["Trust-weighted (oracle feedback) - upper bound", orac_acc, all_agent_tokens])

# ------------------------- PRINT + SAVE ------------------------------
table = pd.DataFrame(results, columns=["Method", "Accuracy", "Avg tokens per question"])
table["Accuracy"] = (table["Accuracy"] * 100).round(1)
table["Avg tokens per question"] = table["Avg tokens per question"].round(0).astype(int)

print(f"===== RESULTS ({TAG}) =====")
print(table.to_string(index=False))

print("\nFinal trust scores (consensus feedback):")
for a, t in cons_trust.items():
    print(f"  {a:15s} {t:.3f}")
print("\nFinal trust scores (oracle feedback):")
for a, t in orac_trust.items():
    print(f"  {a:15s} {t:.3f}")

table.to_csv(f"results_{TAG}.csv", index=False)
print(f"\nSaved: results_{TAG}.csv, trust_history_{TAG}_consensus.csv, "
      f"trust_history_{TAG}_oracle.csv")

# =====================================================================
# PHASE 4: debate.py  -- the "multi-agent debate" baseline
#
# Round 1 = the answers already saved in answers.csv (no re-running!)
# Round 2 = each agent sees the OTHER agents' round-1 solutions and
#           solves the problem again (it can keep or change its answer).
# Final answer = majority vote of the round-2 answers.
#
# Debate costs more tokens (two rounds), which is exactly what we want
# to compare against our trust-weighted method.
#
# Run ONLY after collect_answers.py has finished:
#   python debate.py 5     -> quick test with 5 questions
#   python debate.py       -> full run (about 60-75 minutes)
# Ctrl + C to stop; run again to continue where it stopped.
# =====================================================================

import os
import re
import sys
import csv
import time
import random
import pandas as pd
import ollama
from datasets import load_dataset

# ------------------------- SETTINGS ----------------------------------
AGENTS = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3:mini"]  # 4 main agents only
NUM_QUESTIONS = 200
SEED = 42
OUTPUT_FILE = "debate_answers.csv"
MAX_CHARS = 600   # show only the last 600 characters of each other agent's
                  # reasoning (keeps prompts short and fast)

if len(sys.argv) > 1:
    NUM_QUESTIONS = int(sys.argv[1])

PROMPT = """Solve this math word problem.

Problem: {question}

Other agents solved it like this (their reasoning may contain mistakes):
{others}

Your own earlier answer was: {own}

Check the other solutions carefully, then solve the problem again step by step.
At the very end, write the final answer on its own line in this exact format:
#### <number>"""


# ------------------------- HELPERS (same as Phase 2) -----------------
def find_numbers(text):
    text = text.replace(",", "").replace("$", "")
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def extract_answer(response):
    if "####" in response:
        nums = find_numbers(response.split("####")[-1])
        if nums:
            return float(nums[0])
    nums = find_numbers(response)
    return float(nums[-1]) if nums else None


# ------------------------- LOAD DATA ---------------------------------
print("Loading GSM8K and round-1 answers...")
data = load_dataset("openai/gsm8k", "main", split="test")
round1 = pd.read_csv("answers.csv")
round1 = round1[round1["agent"].isin(AGENTS)]

# Same question order as Phase 2
order = list(range(len(data)))
random.Random(SEED).shuffle(order)
order = order[:NUM_QUESTIONS]

# Keep only questions where all 4 agents have a round-1 answer
r1 = {(r["agent"], r["q_id"]): r for _, r in round1.iterrows()}
order = [q for q in order if all((a, q) in r1 for a in AGENTS)]
print(f"{len(order)} questions ready for debate.")

# ------------------------- RESUME SUPPORT ----------------------------
done = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add((r["agent"], int(r["q_id"])))
    print(f"Found {len(done)} round-2 answers already saved. Continuing.")

new_file = not os.path.exists(OUTPUT_FILE)
out = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(out)
if new_file:
    writer.writerow(["agent", "q_id", "correct_answer", "round1_answer",
                     "round2_answer", "is_correct", "tokens", "seconds", "response"])

# ------------------------- ROUND 2 -----------------------------------
labels = ["Agent A", "Agent B", "Agent C"]  # hide model names so no agent is biased

for agent in AGENTS:
    print(f"\n========== Debate round 2: {agent} ==========")
    right, total = 0, 0
    for n, q in enumerate(order, start=1):
        if (agent, q) in done:
            continue

        own = r1[(agent, q)]
        others_text = ""
        for label, other in zip(labels, [a for a in AGENTS if a != agent]):
            o = r1[(other, q)]
            reasoning = str(o["response"])[-MAX_CHARS:]
            others_text += f"\n--- {label} (final answer: {o['predicted_answer']}) ---\n...{reasoning}\n"

        prompt = PROMPT.format(question=data[q]["question"], others=others_text,
                               own=own["predicted_answer"])
        try:
            start = time.time()
            resp = ollama.chat(model=agent,
                               messages=[{"role": "user", "content": prompt}],
                               options={"temperature": 0, "seed": SEED, "num_predict": 512})
            seconds = round(time.time() - start, 2)
        except Exception as e:
            if "repeat" in str(e).lower():
                # The model got stuck in a loop. Retrying gives the same loop
                # (temperature 0), so record it as "no answer" = counted wrong.
                print(f"  Question {q}: model got stuck in a loop -> recorded as no answer (wrong)")
                writer.writerow([agent, q, float(own["correct_answer"]), own["predicted_answer"],
                                 None, 0, 0, 0, f"FAILED: {e}"])
                out.flush()
                total += 1
            else:
                # Other errors (e.g. Ollama not running): skip, retry next time
                print(f"  Error on question {q}: {e}")
            continue

        text = resp.message.content
        pred = extract_answer(text)
        true_ans = float(own["correct_answer"])
        is_correct = pred is not None and abs(pred - true_ans) < 1e-4
        tokens = (resp.prompt_eval_count or 0) + (resp.eval_count or 0)

        writer.writerow([agent, q, true_ans, own["predicted_answer"], pred,
                         int(is_correct), tokens, seconds, text])
        out.flush()
        total += 1
        right += int(is_correct)
        print(f"  [{n}/{len(order)}] q{q}: round1={own['predicted_answer']} -> round2={pred}, "
              f"true={true_ans}, {'RIGHT' if is_correct else 'wrong'} ({seconds}s) | "
              f"so far {right}/{total}")

out.close()

# ------------------------- DEBATE RESULT -----------------------------
r2 = pd.read_csv(OUTPUT_FILE)
r2map = {(r["agent"], r["q_id"]): r for _, r in r2.iterrows()}
complete = [q for q in order if all((a, q) in r2map for a in AGENTS)]

rng = random.Random(SEED)
right, tok = 0, 0
for q in complete:
    votes = {}
    for a in AGENTS:
        p = r2map[(a, q)]["round2_answer"]
        if not pd.isna(p):
            p = round(float(p), 4)
            votes[p] = votes.get(p, 0) + 1
    final = None
    if votes:
        top = max(votes.values())
        final = rng.choice(sorted(p for p, v in votes.items() if v == top))
    right += int(final == round(float(r2map[(AGENTS[0], q)]["correct_answer"]), 4))
    # debate cost = round-1 tokens + round-2 tokens of all agents
    tok += sum(r1[(a, q)]["tokens"] + r2map[(a, q)]["tokens"] for a in AGENTS)

if complete:
    acc = round(100 * right / len(complete), 1)
    avg_tok = round(tok / len(complete))
    print(f"\n===== DEBATE RESULT on {len(complete)} questions =====")
    print(f"Accuracy: {acc}%   |   Avg tokens per question: {avg_tok}")
    print("\nRound-2 accuracy of each agent (after seeing others):")
    for a in AGENTS:
        acc_a = 100 * sum(r2map[(a, q)]["is_correct"] for q in complete) / len(complete)
        print(f"  {a:15s} {acc_a:.1f}%")
    pd.DataFrame([["Debate (2 rounds, majority vote)", acc, avg_tok]],
                 columns=["Method", "Accuracy", "Avg tokens per question"]
                 ).to_csv("debate_results.csv", index=False)
    print("\nSaved: debate_answers.csv, debate_results.csv")

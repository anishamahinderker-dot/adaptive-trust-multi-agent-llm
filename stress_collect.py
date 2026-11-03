# =====================================================================
# STRESS TEST - step 1: stress_collect.py
# Runs the weak model (qwen2.5:0.5b) 3 more times WITH SAMPLING
# (temperature 0.7, seeds 1, 2, 3). Each run acts as one more
# unreliable agent. Same weak model -> they tend to make similar
# mistakes, like redundant or colluding weak agents.
#
# Answers are added to answers.csv as agents named:
#   qwen2.5:0.5b#s1, qwen2.5:0.5b#s2, qwen2.5:0.5b#s3
#
# Run:  python stress_collect.py     (about 10-15 minutes, tiny model)
# Ctrl + C to stop; run again to continue.
# =====================================================================

import os
import re
import csv
import time
import random
import ollama
from datasets import load_dataset

WEAK_MODEL = "qwen2.5:0.5b"
SAMPLE_SEEDS = [1, 2, 3]   # one extra unreliable agent per seed
TEMPERATURE = 0.7          # sampling -> each run gives slightly different answers
NUM_QUESTIONS = 200
SEED = 42                  # question order seed (same as Phase 2)

import sys
if len(sys.argv) > 1:      # e.g. python stress_collect.py 1319
    NUM_QUESTIONS = int(sys.argv[1])
OUTPUT_FILE = "answers.csv"

PROMPT = """Solve this math word problem step by step.
At the very end, write the final answer on its own line in this exact format:
#### <number>

Problem: {question}"""


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


print("Loading GSM8K...")
data = load_dataset("openai/gsm8k", "main", split="test")
order = list(range(len(data)))
random.Random(SEED).shuffle(order)
order = order[:NUM_QUESTIONS]

done = set()
with open(OUTPUT_FILE, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        done.add((r["agent"], int(r["q_id"])))

out = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(out)

for s in SAMPLE_SEEDS:
    name = f"{WEAK_MODEL}#s{s}"
    print(f"\n========== Unreliable agent: {name} ==========")
    right, total = 0, 0
    for n, q in enumerate(order, start=1):
        if (name, q) in done:
            continue
        row = data[q]
        true_ans = float(row["answer"].split("####")[-1].strip().replace(",", ""))
        try:
            start = time.time()
            resp = ollama.chat(model=WEAK_MODEL,
                               messages=[{"role": "user", "content": PROMPT.format(question=row["question"])}],
                               options={"temperature": TEMPERATURE, "seed": s, "num_predict": 512})
            seconds = round(time.time() - start, 2)
        except Exception as e:
            print(f"  Error on question {q}: {e}")
            continue
        text = resp.message.content
        pred = extract_answer(text)
        ok = pred is not None and abs(pred - true_ans) < 1e-4
        tokens = (resp.prompt_eval_count or 0) + (resp.eval_count or 0)
        writer.writerow([name, q, true_ans, pred, int(ok), tokens, seconds, text])
        out.flush()
        total += 1
        right += int(ok)
        if n % 20 == 0:
            print(f"  [{n}/{NUM_QUESTIONS}] accuracy so far: {right}/{total}")
    if total:
        print(f"  >> {name} finished: {right}/{total} correct")

out.close()
print("\nDone! Now run: python stress_test.py")

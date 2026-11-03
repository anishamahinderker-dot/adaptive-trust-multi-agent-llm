# =====================================================================
# PHASE 2: collect_answers.py
# Every agent (model) answers the SAME GSM8K math questions.
# All answers are saved in answers.csv, so we never have to re-run
# the models again. Phases 3-6 will only read this file.
#
# How to run:
#   python collect_answers.py 5      -> quick test with 5 questions
#   python collect_answers.py        -> full run with 200 questions
#
# You can stop anytime with Ctrl + C. When you run it again, it skips
# the questions already answered and continues from where it stopped.
# =====================================================================

import os
import re
import sys
import csv
import time
import random
import ollama
from datasets import load_dataset

# ------------------------- SETTINGS ----------------------------------
# The 4 main agents + 1 tiny "bad agent" (qwen2.5:0.5b) for the robustness test
AGENTS = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3:mini", "qwen2.5:0.5b"]

NUM_QUESTIONS = 200          # how many GSM8K questions to use
SEED = 42                    # fixed seed = same questions every time (reproducible)
OUTPUT_FILE = "answers.csv"  # where all answers are saved

# If you type a number after the script name, use that instead (for testing)
if len(sys.argv) > 1:
    NUM_QUESTIONS = int(sys.argv[1])

# The instruction every agent gets. Same prompt for all agents = fair comparison.
PROMPT = """Solve this math word problem step by step.
At the very end, write the final answer on its own line in this exact format:
#### <number>

Problem: {question}"""


# ------------------------- HELPER FUNCTIONS --------------------------
def find_numbers(text):
    """Return all numbers found in a piece of text, e.g. '$1,200.5' -> ['1200.5']."""
    text = text.replace(",", "").replace("$", "")
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def extract_answer(response):
    """Pull the final numeric answer out of a model's response.
    First look after '####'. If the model forgot '####', use the last number."""
    if "####" in response:
        nums = find_numbers(response.split("####")[-1])
        if nums:
            return float(nums[0])
    nums = find_numbers(response)
    if nums:
        return float(nums[-1])
    return None  # the model gave no number at all


def correct_answer(row):
    """GSM8K stores the true answer after '####' in the 'answer' field."""
    return float(row["answer"].split("####")[-1].strip().replace(",", ""))


# ------------------------- LOAD QUESTIONS ----------------------------
print("Loading GSM8K...")
data = load_dataset("openai/gsm8k", "main", split="test")

# Shuffle all question numbers with a fixed seed, then take the first N.
# Same seed -> same questions every run. The 5-question test uses the
# first 5 of the same 200, so the test answers are reused later.
question_ids = list(range(len(data)))
random.Random(SEED).shuffle(question_ids)
question_ids = question_ids[:NUM_QUESTIONS]

# ------------------------- RESUME SUPPORT ----------------------------
# Read what's already saved so we don't redo work after a stop/crash.
done = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add((r["agent"], int(r["q_id"])))
    print(f"Found {len(done)} answers already saved. Continuing from there.")

# Open the CSV in "append" mode and write the header only if it's a new file
new_file = not os.path.exists(OUTPUT_FILE)
out = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(out)
if new_file:
    writer.writerow(["agent", "q_id", "correct_answer", "predicted_answer",
                     "is_correct", "tokens", "seconds", "response"])

# ------------------------- MAIN LOOP ---------------------------------
# One agent at a time through ALL questions (your 4 GB GPU holds one
# model at a time, so switching models every question would be slow).
for agent in AGENTS:
    print(f"\n========== Agent: {agent} ==========")
    right = 0
    total = 0

    for n, q_id in enumerate(question_ids, start=1):
        if (agent, q_id) in done:
            continue  # already answered earlier

        row = data[q_id]
        true_ans = correct_answer(row)

        try:
            start = time.time()
            resp = ollama.chat(
                model=agent,
                messages=[{"role": "user", "content": PROMPT.format(question=row["question"])}],
                options={"temperature": 0, "seed": SEED, "num_predict": 512},
            )
            seconds = round(time.time() - start, 2)
        except Exception as e:
            # Skip this one; it will be retried next time you run the script
            print(f"  Error on question {q_id}: {e}")
            continue

        text = resp.message.content
        pred = extract_answer(text)
        is_correct = pred is not None and abs(pred - true_ans) < 1e-4
        tokens = (resp.prompt_eval_count or 0) + (resp.eval_count or 0)

        writer.writerow([agent, q_id, true_ans, pred, int(is_correct), tokens, seconds, text])
        out.flush()  # save to disk immediately, so nothing is lost if it stops

        total += 1
        right += int(is_correct)
        print(f"  [{n}/{NUM_QUESTIONS}] q{q_id}: predicted={pred}, true={true_ans}, "
              f"{'RIGHT' if is_correct else 'wrong'} ({seconds}s) | "
              f"accuracy so far: {right}/{total}")

    if total > 0:
        print(f"  >> {agent} finished: {right}/{total} correct this session")

out.close()
print(f"\nAll done! Answers saved in {OUTPUT_FILE}")

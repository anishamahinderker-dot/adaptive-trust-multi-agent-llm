# Adaptive Trust-Based Multi-Agent LLM Framework

Code, model outputs and analysis scripts for the paper:

> M. Kulkarni, D. D. Patil, G. Kundalwal, A. Mahinderker. *Adaptive Trust-Based Multi-Agent Large Language Model Framework for Reliable Collaborative Reasoning.* 5th International Conference on Data Science and Management (ICDSM 2026), Springer LNNS.

Several LLM agents answer the same problem, and their answers are combined by **trust-weighted voting**. Each agent's trust is an exponential moving average of feedback received after every task:

```
T_i(new) = λ · T_i(old) + (1 − λ) · c_i        (T_i start = 0.5, λ = 0.9)
```

- **Consensus feedback** (label-free): `c_i = 1` if the agent agreed with the final answer.
- **Verification feedback**: `c_i = 1` if the agent was correct (used only after the vote).

## Setup

- **Agents:** Qwen2.5-3B, Llama 3.2-3B, Gemma 2-2B, Phi-3-mini; unreliable agents: Qwen2.5-0.5B
- **Benchmark:** GSM8K test set (1,319 problems)
- **Baselines:** single agent, majority vote, two-round debate (200-problem subset)
- **Hardware used:** AMD Ryzen 7 7435HS, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB)
- **Software:** Python 3.11, Ollama 0.40.13

## Main results (1,319 problems)

| Method | Accuracy (%) |
| --- | --- |
| Best single agent (Phi-3-mini) | 83.4 |
| Majority vote | 85.7 |
| Trust-weighted, consensus feedback | 86.4 |
| Trust-weighted, verification feedback | 86.3 |

With 4 unreliable agents added: majority vote 81.7%, trust-weighted (consensus) 85.2% (exact McNemar test, p < 0.0001).
Two-round debate (200 problems): 87.5% at 5,427 tokens per problem, versus 88.0% at 1,357 tokens for trust-weighted voting.

## How to reproduce

1. Install [Ollama](https://ollama.com) and pull the models:
   ```
   ollama pull qwen2.5:3b
   ollama pull llama3.2:3b
   ollama pull gemma2:2b
   ollama pull phi3:mini
   ollama pull qwen2.5:0.5b
   ```
2. Install the Python libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run the experiment (the saved CSV files let you skip straight to step 4):
   ```
   python collect_answers.py 1319     # all agents answer all problems
   python stress_collect.py 1319      # extra unreliable agents
   python debate.py                   # debate baseline (200 problems)
   ```
4. Compute results, significance tests and figures:
   ```
   python compute_results.py
   python compute_results.py bad
   python compute_results.py 200
   python compute_results.py bad 200
   python stress_test.py
   python make_plots.py
   ```

All scripts resume automatically if stopped, and use fixed seeds so the problem order is identical on every run.

## Files

| File | Contents |
| --- | --- |
| `collect_answers.py` | Every agent answers the GSM8K problems; saves `answers.csv` |
| `stress_collect.py` | Extra sampled unreliable agents for the stress test |
| `debate.py` | Two-round debate baseline; saves `debate_answers.csv` |
| `compute_results.py` | Single-agent, majority-vote and trust-weighted results |
| `stress_test.py` | Accuracy with 0–4 unreliable agents, with McNemar tests |
| `make_plots.py` | Figures and the significance table |
| `run_overnight.bat` | Runs the full 1,319-problem collection on Windows |
| `answers.csv`, `debate_answers.csv` | Raw model outputs |
| `results_*.csv`, `stress_results.csv`, `significance.csv` | Final result tables |
| `figures/` | Figures used in the paper |

## Authors

Mrugakshi Kulkarni, Dipti D. Patil, Gayatri Kundalwal, Anisha Mahinderker — MKSSS's Cummins College of Engineering for Women, Pune, India.

## License

Code released under the MIT License. GSM8K is released by OpenAI under the MIT License; model outputs are subject to the licences of the respective models.

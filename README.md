# HeuriForge

[![CI](https://github.com/TMW17/heuriforge/actions/workflows/ci.yml/badge.svg)](https://github.com/TMW17/heuriforge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**LLM-guided evolution of vehicle-routing heuristics, on one laptop, for $0.**

HeuriForge is a compact, carefully evaluated take on the idea behind AlphaEvolve and
FunSearch: a language model proposes changes to a program, an evolutionary search keeps
the changes that score better, and repeats. Here the model is small and local
(Qwen3.5-4B served by llama.cpp on a MacBook), and the programs are heuristics for the
capacitated vehicle routing problem (CVRP).

What it is built to answer honestly: *how much does a small local model actually add to
evolutionary program search?* So every result is measured the way a reviewer would ask
for it: model-written code runs in a sandbox, an identical search **without** a language
model is the baseline, every configuration is run three times with different search
seeds, the champion of each run is chosen on a validation split, and it is scored once on
a **sealed** test split, with paired statistics.

## Results

Each of the four configurations was evolved three times (search seeds 0, 1 and 2) on a
MacBook Pro (Apple Silicon, 16 GB), 500 candidates per run. The test split stayed sealed
until all twelve runs had finished; then each run's champion was scored once on it.
Full report (charts, champion code, lineage, audit):
[tmw17.github.io/heuriforge](https://tmw17.github.io/heuriforge/).

| Task | Starting point | No-LLM baseline | LLM | LLM vs no-LLM (per test item) |
|---|---|---|---|---|
| 1 · Constructive heuristic | 24.62% | 21.83% ± 2.15 (3 runs) | 17.92% ± 1.80 (3 runs) | -3.91 pts [-5.54, -2.28], p = 0.00021, A12 = 0.62 |
| 2 · PyVRP acceptance rule | 2.72% | 1.78% ± 0.10 (3 runs) | 1.76% ± 0.18 (3 runs) | -0.03 pts [-0.11, +0.06], p = 0.51, A12 = 0.52 |

*Mean gap (%) to the best-known solutions on the sealed test items (Task 1: 20 instances;
Task 2: 20 instances × 5 seeds), as mean ± standard deviation over the three runs of each
configuration. Last column: the champions' gaps are averaged over the runs of each kind,
item by item, then compared: mean difference with a 95% bootstrap confidence interval,
two-sided Wilcoxon signed-rank p-value, and Vargha-Delaney A12 (the probability that the
LLM side does better on an item; 0.5 = no difference). Negative differences favour the
LLM. Task 1 reference on the same test instances: Clarke-Wright savings (1964) 6.54%.*

| Run | Seed | Champion's test gap | Champion vs starting point |
|---|---|---|---|
| `construct-llm` | 0 | 19.79% | -4.83 pts, p = 0.07 (Holm) |
| `construct-llm-s1` | 1 | 16.20% | -8.43 pts, p = 0.0027 (Holm) |
| `construct-llm-s2` | 2 | 17.76% | -6.86 pts, p = 0.0064 (Holm) |
| `construct-random` | 0 | 19.34% | -5.28 pts, p = 0.053 (Holm) |
| `construct-random-s1` | 1 | 23.06% | -1.57 pts, p = 0.43 (Holm) |
| `construct-random-s2` | 2 | 23.08% | -1.54 pts, p = 0.67 (Holm) |
| `accept-llm` | 0 | 1.88% | -0.85 pts, p = 8.6e-14 (Holm) |
| `accept-llm-s1` | 1 | 1.84% | -0.88 pts, p = 8.6e-13 (Holm) |
| `accept-llm-s2` | 2 | 1.55% | -1.17 pts, p = 9.3e-17 (Holm) |
| `accept-random` | 0 | 1.67% | -1.06 pts, p = 8.8e-15 (Holm) |
| `accept-random-s1` | 1 | 1.84% | -0.88 pts, p = 2e-13 (Holm) |
| `accept-random-s2` | 2 | 1.84% | -0.88 pts, p = 2e-13 (Holm) |

What this shows:

- **Task 1: the model helps.** The three model runs improved the starting point by 4.8,
  8.4 and 6.9 points on the test instances (the first one not significantly, p = 0.07).
  Averaged over runs, the model's champions beat the no-model champions by 3.9 points
  (p = 0.0002). Two of the three no-model runs barely moved (about 1.5 points, not
  significant). The evolved heuristics remain far from Clarke-Wright savings (6.54% on
  the same instances): a rule that picks one stop at a time cannot express a merge
  algorithm. The best champion starts each route at a far customer with a small demand
  and extends it by savings; another one picks the first customer of every route through
  an accidental division by zero.
- **Task 2: large gains, but the model adds nothing.** Every champion, with or without
  the model, cuts PyVRP's gap at 1,000 iterations from 2.72% to 1.55-1.88% (p < 10⁻¹²
  each), a third smaller on average. The main discovery is to be greedier at this short
  budget: two no-model runs ended with plain hill climbing ("accept only if not worse"),
  one mutation away from the starting rule. The model's champions add an early random
  walk or a small tolerance for worse solutions; on average they do no better (-0.03
  points, p = 0.51).
- **Cost.** A run with the model used 1.7 to 3.8 hours of model time (two requests in
  parallel: about 1 to 2 hours of wall time); a run without it took 3 to 16 minutes.
- **What small-model code looks like.** 19-30% of the model's Task 1 programs crashed
  (array-shape errors, undefined names), against 4-5% in Task 2, whose code is scalar
  logic. The ideas the model states often do not match the code it writes.

Reference points measured with this repository (training / validation instances):

| | train | validation |
|---|---|---|
| Task 1 starting point: nearest neighbour | 31.53% | 29.69% |
| Task 1 reference: Clarke-Wright savings (1964) | 6.80% | 5.97% |
| Task 2 starting point: PyVRP's own rule, 1,000 iterations | 2.75% | 2.83% |

## The two tasks

**Task 1: a constructive heuristic** (in the style of EoH and ReEvo). A fixed loop builds
the routes one stop at a time; the evolved function scores the possible next stops
(`score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot)`).
It starts as nearest neighbour. It is fast to evaluate, so the signal is strong. A one-stop-at-a-time
rule cannot express merge-based algorithms, so Clarke-Wright savings is reported as an
outside reference, not as a target.

**Task 2: the acceptance rule inside a state-of-the-art solver.** PyVRP 0.14 solves the CVRP
with iterated local search, and after every step it decides whether to accept the new
solution (late-acceptance hill climbing). HeuriForge replicates PyVRP's loop step for step
with that single decision made pluggable
(`accept(cand, curr, late, best, feasible, it, max_it, stall, u, state)`); a test checks that
the replica with the original rule reproduces `pyvrp.solve` exactly. Gains here are small
and hard-won, which is the point.

## How it works

```mermaid
flowchart LR
    A["Archive<br/>2 islands × MAP-Elites grid"] -->|parent + inspirations| P[Prompt]
    B["Bandit<br/>model × mutation"] --> P
    L[Lessons] --> P
    P --> M["Qwen3.5-4B<br/>llama-server, JSON grammar"]
    M -->|edits or rewrite| C[Child program]
    C --> S{Static checks}
    S -->|unsafe| X[Rejected and audited]
    S --> N{Novel?}
    N -->|duplicate| X
    N --> SM["Smoke test ×2<br/>(sandbox)"]
    SM --> TR["Training score<br/>(sandbox, routes re-verified)"]
    TR --> A
    TR -->|new island best| V[Validation score]
    V --> CH[Champion] --> T[Sealed test, once]
```

- **Search.** Two islands, each with a MAP-Elites grid over code size and the number of
  inputs a program uses, so different *kinds* of good programs survive. Parents are drawn
  as in ShinkaEvolve (fitness-weighted, discounted by how many children a program already
  has); islands exchange their best program in a ring. A cost-aware Thompson-sampling
  bandit picks the model and the mutation type (small SEARCH/REPLACE edits, or a full
  rewrite). Every 25 candidates the model writes a one-line lesson from a better and a
  worse program (ReEvo-style reflection), and later prompts include the latest lessons.
- **The model.** Qwen3.5-4B (Q4_K_M, 2.7 GB) in llama.cpp's `llama-server`, with Metal on
  Apple Silicon. Thinking is turned off. Small edits come back as JSON constrained by a
  schema that llama.cpp compiles to a grammar. Full rewrites come back as a two- or
  three-sentence plan followed by a plain Python code block: inside a JSON string the
  model often never closed the string, and without a plan it reasoned in long code
  comments until it hit the token limit. Edits with a wrong indentation are re-indented
  to the lines they replace when that makes the code compile. The fixed part
  of the prompt sits in the system message, where llama.cpp checkpoints the state of this
  hybrid model, so it can be reused from one candidate to the next instead of being
  recomputed. Requests can only go to localhost.
- **The no-model baseline.** The same search, with the language model replaced by random
  code mutations: change a constant, add a term from a short hand-written list, swap an
  operator, or drop a line. The list carries domain knowledge (for example the savings
  term `dist[current, depot] - dist[candidates, depot]` in Task 1, or a
  simulated-annealing test in Task 2), so this is a strong baseline, not blind random
  search. Both searches get the same number of candidates.
- **Novelty.** Programs that differ only in names, comments or formatting are rejected
  before evaluation (canonical syntax-tree hash); variants that only change constants are
  capped, to push the search towards new ideas.
- **Evaluation integrity.**
  - Static checks reject forbidden imports, file and introspection access, dunder tricks,
    global state and signature changes. The worker runs the same checks again.
  - Every evaluation runs in a fresh process with restricted builtins, a guarded import,
    read-only inputs, a CPU timer per instance, a CPU-time limit, no file writes, a memory
    limit (address space on Linux, a peak-memory watchdog on macOS), a clean environment,
    and its whole process group is killed on timeout.
  - The candidate never reports a score: the parent process re-checks every route (each
    customer once, no overloaded truck) and recomputes the cost itself.
  - The smoke test runs every candidate twice on the same input; different answers mean
    hidden state, and the candidate is rejected.
  - Every rejection has a category, counted per run: the **reward-hacking audit**.
- **Statistics.** Each champion is compared with the starting point and the baseline on the
  same test items: Wilcoxon signed-rank test with Holm's correction, Vargha-Delaney A12
  effect size, and a bootstrap 95% confidence interval of the mean gap difference. To
  compare the model with the no-model baseline, the champions' gaps are averaged over the
  three runs of each configuration, item by item, and compared the same way; the spread
  between runs is reported next to it. The sealed test refuses to run twice unless
  forced, and a forced re-run is recorded.

**Data.** The 50 CVRPLIB set X instances with 100-330 customers (Uchoa et al., 2017) with
their best-known solutions, split 20 / 10 / 20 into train / validation / test, stratified
by average route length so that every split covers short and long routes. Small synthetic
instances are generated for the smoke test, the offline demo and CI.

## Quick start

On macOS with [Homebrew](https://brew.sh) (see [BUILD_GUIDE.md](BUILD_GUIDE.md) for the
full step-by-step version, including the benchmark and publishing the results):

```bash
brew install uv llama.cpp
git clone https://github.com/TMW17/heuriforge.git ~/Projects/heuriforge
cd ~/Projects/heuriforge
make setup        # Python 3.12 + dependencies + git hooks
make test         # about 100 tests, including sandbox-escape attempts
make demo         # offline end-to-end run with a mock model (report in site-demo/)
make data         # CVRPLIB X instances + synthetic ones
make server       # terminal 2: local model server (first run downloads the model)
make bench        # Phase 0: is the model fast and reliable enough?
make construct-llm && make construct-random   # Task 1: with and without the model
make accept-llm && make accept-random         # Task 2: with and without the model
make repeats-llm  # seeds 1 and 2 of both model runs (and make repeats-random)
make sealed-tests # the sealed test of every finished run, once
make table        # the results table, pooled over the runs
make report       # results website in site/
```

| Command | What it does |
|---|---|
| `make eval` | scores both starting programs (sanity check) |
| `make status RUN=…` | progress of a run, also while it runs |
| `make construct-llm RESUME=1` | continues a stopped run (Ctrl+C or a `STOP` file stop it safely) |
| `make construct-llm SEED=1` | a repeated run with another search seed (folder `runs/construct-llm-s1`) |
| `make sealed-test RUN=…` | the sealed test of one run |
| `make export-all` | copies every run's champion and results to `results/` for committing |
| `uv run heuriforge --help` | every command and option |

All settings (model server, search parameters, budgets, time limits) are in
[`config/settings.yaml`](config/settings.yaml). Optional MLflow tracking:
`uv sync --extra tracking` and `tracking.mlflow: true`.

## Repository layout

```
config/settings.yaml            every setting, commented
config/splits/                  which instances are train / validation / test (test is sealed)
src/heuriforge/
  tasks/                        the two tasks: evolvable function, trusted skeleton, prompt text
  sandbox/                      static checks, restricted execution, the worker process
  engine.py                     the evolution loop
  archive.py, bandit.py         parent selection (MAP-Elites, islands) and arm selection
  prompts.py, llm.py            prompts, JSON schemas, llama-server client and offline mock
  mutators.py, novelty.py       random-mutation baseline, duplicate detection
  evaluator.py, instances.py    scoring, route checking, instance download and generation
  sealed_test.py, stats.py      the one-time test and its paired statistics
  report.py, bench.py, cli.py   results website, Phase 0 benchmark, command line
tests/                          unit and end-to-end tests, sandbox-escape and cheating attempts
```

## Limitations

- A 4B model writes weaker code than frontier models; the results describe what a small
  local model can do, not the ceiling of the method.
- Each configuration is evolved three times. That shows how much a result depends on the
  run, but three runs are too few for a significance test at the level of runs: the
  paired tests use the test instances as the sample, with the gaps averaged over runs.
- The no-model baseline draws from hand-written building blocks (see above), so the
  comparison measures what the model adds over a search that already has some domain
  knowledge, not over blind random search.
- Task 2 uses a fixed budget of 1,000 iterations; a rule tuned for short runs may not
  help long ones.
- The sandbox is designed for code written by a model asked to write heuristics, not for
  hostile code from the internet. On macOS the memory limit is a best-effort watchdog.

## References

- Novikov et al. (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* arXiv:2506.13131.
- Romera-Paredes et al. (2024). *Mathematical discoveries from program search with large language models.* Nature 625.
- Liu et al. (2024). *Evolution of Heuristics: Towards efficient automatic algorithm design using large language model.* ICML.
- Ye et al. (2024). *ReEvo: Large language models as hyper-heuristics with reflective evolution.* NeurIPS.
- Lange, Imajuku and Cetin (2025). *ShinkaEvolve: Towards open-ended and sample-efficient program evolution.* arXiv:2509.19349.
- Mouret and Clune (2015). *Illuminating search spaces by mapping elites.* arXiv:1504.04909.
- Wouda, Lan and Kool (2024). *PyVRP: A high-performance VRP solver package.* INFORMS Journal on Computing.
- Uchoa et al. (2017). *New benchmark instances for the capacitated vehicle routing problem.* European Journal of Operational Research 257(3).
- Burke and Bykov (2017). *The late acceptance hill-climbing heuristic.* European Journal of Operational Research 258(1).
- Clarke and Wright (1964). *Scheduling of vehicles from a central depot to a number of delivery points.* Operations Research 12(4).
- Vargha and Delaney (2000). *A critique and improvement of the CL common language effect size statistics of McGraw and Wong.* Journal of Educational and Behavioral Statistics 25(2).

## License

MIT. The CVRPLIB X instances are downloaded at run time from
[PyVRP/Instances](https://github.com/PyVRP/Instances) and are not redistributed here (one
instance is included under `tests/data` for the tests).

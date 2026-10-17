# HeuriForge: step-by-step guide (MacBook Pro, Apple Silicon)

This guide takes you from the zip file to published results: tests, an offline demo, the
local model server, the Phase 0 benchmark, four evolution runs and their two repeats with
other seeds, the sealed tests, the results website on GitHub Pages and the README table.

Every command goes in the **Terminal** app. Lines starting with `#` are comments; you do
not type them. When a step says *expected*, compare with what you see before moving on.

**Time.** About 30 minutes of your own time, plus 10 to 12 hours of runs that work alone
(leave the Mac running overnight). The real durations depend on your Mac; step 7
measures them.

---

## 1. Check the tools

You already installed Homebrew, uv, git and the GitHub CLI for SahelCast. Check them, and
install llama.cpp (the program that runs the language model on your Mac's GPU):

```bash
brew --version && uv --version && git --version && gh auth status
brew install llama.cpp
llama-server --version
```

*Expected:* four version lines, `Logged in to github.com account TMW17`, and a
`version: ...` line from llama-server.

## 2. Put the project in ~/Projects (not on the Desktop)

Your Desktop is synced with iCloud, and iCloud hides some files Python needs (that broke
SahelCast at first). Use a normal folder:

```bash
mkdir -p ~/Projects
cd ~/Projects
unzip ~/Downloads/heuriforge.zip
cd heuriforge
ls
```

*Expected:* `BUILD_GUIDE.md  LICENSE  Makefile  README.md  config  pyproject.toml  src  tests  uv.lock`.

## 3. Install and test

```bash
make setup
make test
```

`make setup` creates the git repository, installs Python 3.12 and every dependency in
`.venv`, and installs the pre-commit hooks. `make test` runs about 100 tests in about a
minute, including the ones that try to escape the sandbox or cheat the evaluator.

*Expected:* a line like `100 passed in 55.12s`.

## 4. Offline demo (no model, no download)

```bash
make demo
open site-demo/index.html
```

This generates small synthetic instances, runs two short evolutions (a mock "model"
that makes random code changes, and the no-model baseline), runs their sealed tests and
builds a report. It shows the whole pipeline end to end in a minute or two. The numbers
mean nothing (the page says so in a yellow box); you are only checking that everything
works and seeing what the final report will look like.

## 5. Download the benchmark data

```bash
make data
```

It downloads the 50 X instances with their best-known solutions and generates small
synthetic instances (the first run of `make demo` already generated some).

*Expected:* `53 instances in .../data/instances/cvrplib_x` (the 50 X instances and 3 tiny
synthetic ones) and `31 instances in .../data/instances/synthetic`, in about a minute.

## 6. Sanity check: score the two starting programs

```bash
make eval
```

*Expected* (the last lines of each table):

```
mean                  31.53       6.80      <- Task 1: nearest neighbour, and Clarke-Wright savings
mean                   2.75                 <- Task 2: PyVRP's own rule, 1,000 iterations
```

The Task 1 numbers must match exactly. Task 2 should match too; a difference in the
second decimal is possible on a different processor and is not a problem.

## 7. Start the model server and run the Phase 0 benchmark

Open a **second Terminal window** (Cmd+N) and start the server there. It must stay
open during the runs.

```bash
cd ~/Projects/heuriforge
make server
```

The first time, it downloads Qwen3.5-4B (Q4_K_M, 2.7 GB) into `~/Library/Caches/llama.cpp`.
It is ready when you see a line like `server is listening on http://127.0.0.1:8080`.

Back in the **first window**:

```bash
curl -s http://127.0.0.1:8080/health; echo
make bench
```

*Expected:* `{"status":"ok"}`, then a table in this format (the numbers below are only an
illustration; yours are the real ones):

```
in flight cand/hour  s/cand gen tok/s prompt tok/s  cache  parsed usable
        1       180    20.1      38.5          650    55%    100%    83%
        2       290    24.8      32.0          600    55%    100%    67%
```

How to read it:

| Column | Meaning | What you want |
|---|---|---|
| `cand/hour` | candidates the model can propose per hour | with 2 in flight, more than about 100 |
| `s/cand` | seconds per answer | under about 60 |
| `parsed` | answers in the requested format | close to 100% |
| `usable` | answers whose edits apply and pass the safety checks | above about 40% |
| `cache` | share of the prompt reused from the previous request | any value; it only changes speed |

Decisions:
- **`parsed` low**: the line `end of that answer:` under the table shows how a failed
  answer ended. Send it to me if the reason is not obvious.
- **`usable` low**: read the `not usable` lines under the table (`that answer:` shows one
  failed answer; the saved file in `bench/` has all of them). `edit_failed` means the
  model copied the code wrongly (normal now and then); `cut off` means answers are too
  long: raise `llm.max_tokens` in `config/settings.yaml`.
- **Too slow** (fewer than about 60 candidates per hour): stop the server (Ctrl+C) and try
  the smaller model: `make server MODEL=unsloth/Qwen3.5-2B-GGUF:Q4_K_M`, then `make bench`.
  If you keep it, write that model name in `config/settings.yaml` (`llm.models`).
- **A run takes** `max_candidates / cand-per-hour` hours, plus evaluation (larger for
  Task 2). With the default 500 candidates and 300 candidates per hour, Task 1 takes
  about 2 hours.

Each benchmark is saved in `bench/`.

## 8. Publish the code before the runs (seals the test split)

Commit and push now, before any result exists. The commit date proves that the
train / validation / test split in `config/splits/` was fixed before evolution started.

```bash
git add -A
git commit -m "HeuriForge: LLM-guided evolution of VRP heuristics"
```

If the commit stops because a hook reformatted a file, run the same two commands again.

```bash
gh repo create heuriforge --public --source=. --remote=origin --push \
  --description "LLM-guided evolution of vehicle-routing heuristics with a small local model"
gh run watch
```

*Expected:* the CI run (tests on Linux and macOS) ends with a green check.

## 9. Task 1 runs

The server must be running in the second window. In the first window:

```bash
make construct-llm
```

It prints one line per candidate:

```
[  37/500] island 0  qwen3.5-4b/diff          evaluated          train  24.12%  (parent 25.03%)  * island best, validation 22.80%  18.2s
```

- `evaluated`: the program ran on the 20 training instances; `train` is its average gap
  to the best-known solutions (lower is better).
- `* island best`: a new best program on that island; it is also scored on the
  validation instances, which choose the champion.
- Other statuses (`edit_failed`, `rejected_static`, `failed_smoke`, ...) are normal; the
  report counts them.

To follow a run from a **third window** while it runs:

```bash
cd ~/Projects/heuriforge
make status RUN=construct-llm
```

**Stopping and continuing.** Ctrl+C stops safely (everything finished is saved), and so
does creating a file named `STOP` in the run folder (`touch runs/construct-llm/STOP`).
Continue later with `make construct-llm RESUME=1` (delete the `STOP` file first). The Mac
must stay awake: the `make` targets use `caffeinate`, but keep the charger connected and
the lid open.

Then the baseline, which does not need the server (you can stop it to free memory):

```bash
make construct-random
```

## 10. Task 2 runs

Start the server again if you stopped it, then:

```bash
make accept-llm
make accept-random
```

Each candidate takes longer here: PyVRP solves the 20 training instances for 1,000
iterations each. Improvements are small (tenths of a percent); that is expected for a
state-of-the-art solver.

## 11. Repeat every run with two other seeds

One run per configuration can be lucky or unlucky. Each of the four runs is repeated
with search seeds 1 and 2, with the same code and settings; the runs of steps 9 and 10
are seed 0. The model runs need the server, the others do not, so run both groups at the
same time in two windows:

```bash
make repeats-llm      # window 1 (server running): about 6 hours
```

```bash
make repeats-random   # window 2: under an hour
```

The repeats are in `runs/construct-llm-s1`, `runs/accept-llm-s2`, and so on. Follow one
with `make status RUN=construct-llm-s1`. If they stop (Ctrl+C, the Mac went to sleep),
run the same command again: finished runs are skipped and an unfinished one continues.

## 12. The sealed tests (once per run)

Only when all twelve runs are finished (the server is not needed any more):

```bash
make sealed-tests
```

It scores the champion of every finished run (chosen on validation) on the 20 test
instances that evolution never saw (for Task 2, with 5 new seeds each), and compares it
with the starting program, with paired statistics. Runs that are not finished are
skipped; a run that already had its sealed test is never tested again: the point of a
sealed test is that you look at it once. (`make sealed-test RUN=<name>` tests a single
run; `uv run heuriforge test --name <name> --force` exists for a real bug fix, and the
re-run is then recorded and shown in the report.)

How to read a line:

```
champion vs initial: -14.20 points [95% CI -16.9, -11.6], p = 0.000191 (Holm), A12 = 0.93 (large), wins/losses/ties 20/0/0
```

The champion's gap is 14.2 points lower than the starting program's, the true
difference is very likely between 11.6 and 16.9 points, the result is very unlikely to
be luck (p < 0.05), and the champion is better on a random test instance with
probability 0.93. *(Example numbers, not results.)*

## 13. The report and GitHub Pages

```bash
make report
open site/index.html
```

Publish it on the `gh-pages` branch (only the website goes there):

```bash
cd site
git init -b gh-pages
git add -A
git commit -m "Results website"
git push -f https://github.com/TMW17/heuriforge.git gh-pages
rm -rf .git
cd ..
gh api -X POST repos/TMW17/heuriforge/pages -f "source[branch]=gh-pages" -f "source[path]=/"
gh repo edit --homepage https://tmw17.github.io/heuriforge/ \
  --add-topic llm --add-topic evolutionary-algorithms --add-topic vehicle-routing \
  --add-topic llama-cpp --add-topic program-synthesis
```

If the `gh api` command answers with a 409 error, Pages was already enabled: that is
fine. After a minute or two the site is live at https://tmw17.github.io/heuriforge/.
To update it later, run `make report` and the `cd site ... cd ..` block again.

## 14. Fill in the README and commit the results

```bash
make table
```

It prints the results table in Markdown, computed from the twelve sealed tests: for
each task, the starting point, the no-model and the model champions (mean ± standard
deviation over their three runs), and the comparison of the two, item by item; then one
row per run. Replace the table under **Results** in `README.md` with it (any text
editor; for example `open -a TextEdit README.md`), and replace the "Pending" note with
one sentence and the link to the website. Then save the champions and results in the
repository:

```bash
make export-all
git add README.md results
git commit -m "Results: sealed tests of the twelve runs"
git push
```

## 15. If something goes wrong

| Symptom | Fix |
|---|---|
| `llama-server: command not found` | `brew install llama.cpp` |
| `no model server answers at http://127.0.0.1:8080/v1` | Start `make server` in another window and wait for `listening`. |
| `address already in use` when starting the server | A server is already running. `pkill llama-server`, then `make server`. |
| The model download fails or stalls | Run `make server` again; it resumes. Check the internet connection. |
| `this run ... already has a run. Continue it with --resume` | Add `RESUME=1`, or delete `runs/<name>` to start over. |
| A run looks stuck | `make status RUN=<name>` and `tail runs/<name>/log.txt`. The first candidates are slower while the model warms up. |
| The Mac feels slow during runs | Normal. To lighten it, set `eval.workers: 2` in `config/settings.yaml`. |
| `uv` cannot find Python 3.12 | `uv python install 3.12`, then `make setup`. |
| `ModuleNotFoundError: heuriforge` | The project is in an iCloud folder. Move it to `~/Projects`, delete `.venv`, run `make setup`. |
| A commit fails after "files were modified by this hook" | The formatter fixed something. Run `git add -A` and the same `git commit` again. |

## 16. Settings you may want to change

All in `config/settings.yaml` (commented):

- `run.max_candidates` (500) and `run.max_hours` (10): the budget of each run.
- `run.parallel` (2): candidates in flight; matches the server's 2 slots (`SLOTS` in the Makefile).
- `eval.workers` (4): sandbox processes at the same time.
- `llm.models`: add a second server (for example the 2B model on port 8081, started with
  `make server MODEL=unsloth/Qwen3.5-2B-GGUF:Q4_K_M PORT=8081`); the bandit then learns
  which model is worth its cost.
- `tracking.mlflow: true` after `uv sync --extra tracking`: every run is also logged to
  MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`).

Keep the settings identical for a run and its baseline, so the comparison is fair.

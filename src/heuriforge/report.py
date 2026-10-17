"""Static results website (one HTML page, Plotly charts), ready for GitHub Pages.

For every run: progress of the best training gap, what happened to the candidates,
the champion's lineage and its code diff, the sealed-test table, the lessons, the
reward-hacking audit and the bandit's choices. Runs of the same task are also compared
with each other (for example the language model against the random baseline).
"""

from __future__ import annotations

import difflib
import html
import json
import time
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from heuriforge.bandit import Bandit
from heuriforge.database import STATUSES, Database, Program
from heuriforge.engine import champion, run_summary
from heuriforge.stats import compare

STATUS_LABELS = {
    "llm_error": "model call failed",
    "edit_failed": "edit did not apply",
    "rejected_static": "unsafe code rejected",
    "rejected_duplicate": "duplicate rejected",
    "failed_smoke": "failed smoke test",
    "failed_train": "failed on training set",
    "evaluated": "evaluated",
}
AUDIT = {
    "forbidden_import": "imported a forbidden module",
    "forbidden_call": "called a forbidden function",
    "forbidden_attribute": "used a forbidden attribute",
    "forbidden_statement": "used a forbidden statement",
    "dunder_name": "used a double-underscore name",
    "forbidden_name": "used a forbidden name (such as BaseException)",
    "signature_changed": "changed the function signature",
    "too_large": "code too large",
    "syntax_error": "syntax error",
    "modified_input": "tried to modify its read-only input",
    "nondeterministic": "gave different results on the same input",
    "invalid_output": "returned an invalid output",
    "timeout": "too slow (time limit)",
    "memory_limit": "used too much memory",
    "runtime_error": "crashed (exception)",
    "crash": "sandbox process crashed",
}
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]

CSS = """
:root { --fg:#1f2937; --muted:#6b7280; --bg:#ffffff; --card:#f9fafb; --line:#e5e7eb; --accent:#2563eb; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e5e7eb; --muted:#9ca3af; --bg:#0b1020; --card:#111827; --line:#1f2937; --accent:#60a5fa; }
}
* { box-sizing: border-box; }
body { margin:0; font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       color:var(--fg); background:var(--bg); }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.9rem; margin:0.2em 0; } h2 { margin-top:2.2em; border-bottom:1px solid var(--line); padding-bottom:.3em; }
h3 { margin-top:1.6em; } .muted { color:var(--muted); }
.banner { background:#fef3c7; color:#92400e; border:1px solid #f59e0b; padding:10px 14px; border-radius:8px; margin:16px 0; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin:16px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.card .v { font-size:1.45rem; font-weight:650; } .card .k { color:var(--muted); font-size:.85rem; }
table { border-collapse:collapse; width:100%; margin:10px 0; font-size:.92rem; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; } td.n { text-align:right; font-variant-numeric:tabular-nums; }
.scroll { overflow-x:auto; }
pre { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; overflow-x:auto; font-size:.82rem; line-height:1.45; }
.add { color:#16a34a; } .del { color:#dc2626; } .hunk { color:var(--muted); }
details { margin:8px 0; } summary { cursor:pointer; color:var(--accent); }
footer { margin-top:48px; color:var(--muted); font-size:.85rem; }
"""


def esc(x) -> str:
    return html.escape(str(x))


def pct(x, digits: int = 2) -> str:
    return "–" if x is None else f"{x:.{digits}f}%"


def fig_html(fig: go.Figure) -> str:
    fig.update_layout(
        template="plotly_white", margin=dict(l=50, r=20, t=40, b=40), height=380,
        legend=dict(orientation="h", y=-0.2), font=dict(size=13),
    )  # fmt: skip
    return f'<div class="scroll">{fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})}</div>'


# --------------------------------------------------------------------------- data
class RunData:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        db = Database(path / "programs.sqlite", readonly=True)
        self.task = db.get_meta("task")
        self.dataset = db.get_meta("dataset")
        self.provider = db.get_meta("provider")
        self.settings = db.get_meta("settings") or {}
        self.programs = db.programs()
        self.summary = run_summary(db)
        self.champion = champion(db)
        self.lessons = db.lessons()
        self.arms = Bandit(
            {k: v["cost"] for k, v in db.arms().items()} or {"none|none": 1}, db
        ).table()
        self.events = db.events()
        db.close()
        sessions = [e for e in self.events if e["kind"] in ("session_start", "session_end")]
        if sessions and sessions[-1]["kind"] == "session_end":
            self.summary["stop_reason"] = sessions[-1].get("reason") or "interrupted"
        else:
            self.summary["stop_reason"] = "still running (or interrupted)"
        sealed = path / "sealed_test.json"
        self.sealed = json.loads(sealed.read_text()) if sealed.exists() else None
        self.by_id = {p.id: p for p in self.programs}
        self.initial = next((p for p in self.programs if p.origin == "initial"), None)
        self.candidates = [p for p in self.programs if p.origin in ("llm", "random")]

    @property
    def model_label(self) -> str:
        if self.provider == "random":
            return "no language model (random mutations)"
        if self.provider == "mock":
            return "mock model (offline demo)"
        models = self.settings.get("llm", {}).get("models", [])
        return ", ".join(m.get("name", "?") for m in models) or "language model"

    def best_curve(self) -> tuple[list[int], list[float]]:
        best = self.initial.train_score if self.initial else None
        xs, ys = [0], [best]
        for p in sorted(self.candidates, key=lambda p: p.attempt or 0):
            if p.status == "evaluated" and (best is None or p.train_score < best):
                best = p.train_score
            xs.append(p.attempt)
            ys.append(best)
        return xs, ys

    def lineage(self) -> list[Program]:
        chain, p = [], self.champion
        seen = set()
        while p is not None and p.id not in seen:
            seen.add(p.id)
            chain.append(p)
            p = self.by_id.get(p.parent_id) if p.parent_id else None
        return chain[::-1]


# --------------------------------------------------------------------------- sections
def progress_figure(run: RunData) -> str:
    fig = go.Figure()
    arms = sorted({f"{p.model}/{p.mutation}" for p in run.candidates if p.status == "evaluated"})
    initial = run.initial.train_score if run.initial else None
    evaluated = [p for p in run.candidates if p.status == "evaluated"]
    for i, arm in enumerate(arms):
        pts = [p for p in evaluated if f"{p.model}/{p.mutation}" == arm]
        fig.add_trace(
            go.Scatter(
                x=[p.attempt for p in pts],
                y=[p.train_score for p in pts],
                mode="markers",
                name=arm,
                marker=dict(size=6, opacity=0.55, color=PALETTE[i % len(PALETTE)]),
                text=[esc(p.idea or "")[:120] for p in pts],
                hovertemplate="#%{x}: %{y:.2f}%<br>%{text}<extra></extra>",
            )  # fmt: skip
        )
    xs, ys = run.best_curve()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="best so far",
            line=dict(width=3, color="#111827", shape="hv"),
        )
    )
    if initial is not None and ys:
        low = min(y for y in ys if y is not None)
        span = max(initial - low, 0.05 * abs(initial), 0.01)
        fig.update_yaxes(
            range=[low - 0.1 * span, initial + span]
        )  # much worse candidates fall off the top
    fig.update_layout(title="Every evaluated candidate")
    fig.update_xaxes(title="candidate number")
    fig.update_yaxes(title="training gap (%)")
    return fig_html(fig)


def outcome_figure(run: RunData) -> str:
    counts = run.summary["statuses"]
    labels = [STATUS_LABELS[s] for s in STATUSES]
    values = [counts.get(s, 0) for s in STATUSES]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#2563eb",
            text=values,
            textposition="auto",
        )
    )
    fig.update_layout(title="What happened to the candidates", yaxis=dict(autorange="reversed"))
    return fig_html(fig)


def comparison_figure(runs: list[RunData]) -> str:
    fig = go.Figure()
    for i, run in enumerate(runs):
        xs, ys = run.best_curve()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=run.name,
                                 line=dict(width=3, shape="hv", color=PALETTE[i % len(PALETTE)])))  # fmt: skip
    fig.update_layout(title="Best so far, run by run")
    fig.update_xaxes(title="candidate number")
    fig.update_yaxes(title="training gap (%)")
    return fig_html(fig)


def diff_html(old: str, new: str) -> str:
    lines = []
    for line in difflib.unified_diff(
        old.splitlines(), new.splitlines(), "initial", "champion", lineterm=""
    ):
        cls = (
            "add"
            if line.startswith("+")
            else "del"
            if line.startswith("-")
            else "hunk"
            if line.startswith("@@")
            else ""
        )
        lines.append(f'<span class="{cls}">{esc(line)}</span>' if cls else esc(line))
    return "<pre>" + ("\n".join(lines) or "(no change)") + "</pre>"


def table(headers: list[str], rows: list[list], numeric: set[int] = frozenset()) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="n">{c}</td>' if i in numeric else f"<td>{c}</td>" for i, c in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def sealed_html(run: RunData) -> str:
    s = run.sealed
    if not s:
        return '<p class="muted">The sealed test has not been run yet (<code>heuriforge test</code>).</p>'
    forced = " <strong>(forced re-run)</strong>" if s.get("forced") else ""
    rows = [
        [esc(name), pct(gap)] for name, gap in sorted(s["mean_gap"].items(), key=lambda kv: kv[1])
    ]
    out = [
        f'<p class="muted">{s["items"]} test items, never used during evolution{forced}. '
        f"Test split fingerprint <code>{esc(s['test_split_fingerprint'])}</code>.</p>",
        table(["program", "mean test gap"], rows, {1}),
    ]
    if s.get("comparisons"):
        crow = []
        for name, c in s["comparisons"].items():
            crow.append(
                [
                    f"champion vs {esc(name)}",
                    f"{c['mean_diff']:+.2f} pts [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]",
                    f"{c['p_holm']:.3g}",
                    f"{c['a12']:.2f} ({c['effect']})",
                    f"{c['wins']} / {c['losses']} / {c['ties']}",
                ]
            )
        out.append(
            table(
                [
                    "comparison",
                    "mean gap difference [95% CI]",
                    "p (Wilcoxon, Holm)",
                    "A12 (effect)",
                    "wins / losses / ties",
                ],
                crow,
            )
        )
        out.append(
            '<p class="muted">Negative differences mean the champion is better. A12 is the chance that the '
            "champion beats the other program on a random test item (0.5 = no difference).</p>"
        )
    if s.get("errors"):
        out.append("<p>Errors: " + esc(s["errors"]) + "</p>")
    return "\n".join(out)


def audit_html(run: RunData) -> str:
    errors = run.summary["errors"]
    rows = [[esc(AUDIT[k]), f"<code>{esc(k)}</code>", errors[k]] for k in AUDIT if errors.get(k)]
    if not rows:
        return '<p class="muted">No rule violations or crashes.</p>'
    return table(["what the candidate did", "category", "count"], rows, {2})


def run_section(run: RunData) -> str:
    s = run.summary
    champ = run.champion
    cards = [
        ("candidates", s["candidates"]),
        ("evaluated", s["statuses"].get("evaluated", 0)),
        ("initial, validation", pct(s["initial_val_gap"])),
        ("champion, validation", pct(s["champion_val_gap"])),
    ]
    if run.sealed and "champion" in run.sealed["mean_gap"]:
        cards.append(("champion, sealed test", pct(run.sealed["mean_gap"]["champion"])))
    if run.sealed and "initial" in run.sealed["mean_gap"]:
        cards.append(("initial, sealed test", pct(run.sealed["mean_gap"]["initial"])))
    card_html = "".join(
        f'<div class="card"><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div></div>'
        for k, v in cards
    )

    parts = [
        f'<h3 id="{esc(run.name)}">Run <code>{esc(run.name)}</code></h3>',
        f'<p class="muted">{esc(run.task)} on {esc(run.dataset)} · {esc(run.model_label)} · '
        f"status: {esc(s.get('stop_reason'))}</p>",
        f'<div class="cards">{card_html}</div>',
        progress_figure(run),
        '<p class="muted">Each dot is a candidate that ran on the training instances, coloured by '
        "the model and mutation that made it; the black line is the best so far. Lower is better; "
        "much worse candidates are above the chart.</p>",
        outcome_figure(run),
        "<h4>Sealed test</h4>",
        sealed_html(run),
    ]
    if champ is not None and run.initial is not None:
        lineage_rows = [
            [
                p.generation,
                p.id,
                esc(f"{p.model}/{p.mutation}" if p.model else p.origin),
                pct(p.train_score),
                pct(p.val_score),
                esc(p.idea or ("initial program" if p.origin == "initial" else "")),
            ]
            for p in run.lineage()
        ]
        parts += [
            "<h4>How the champion was found</h4>",
            table(
                ["generation", "program", "made by", "train gap", "validation gap", "idea"],
                lineage_rows,
                {0, 1, 3, 4},
            ),
            "<details><summary>The champion's code</summary>",
            f"<pre>{esc(champ.code)}</pre>",
            "</details>",
            "<details><summary>Code changes from the initial program to the champion</summary>",
            diff_html(run.initial.code, champ.code),
            "</details>",
        ]
    if run.lessons:
        items = "".join(f"<li>{esc(t)}</li>" for t in run.lessons[:10])
        parts += ["<h4>Lessons the model wrote</h4>", f"<ul>{items}</ul>"]
    parts += ["<h4>Reward-hacking audit and failures</h4>", audit_html(run)]
    arm_rows = [
        [
            esc(a["model"]),
            esc(a["mutation"]),
            a["pulls"],
            a["successes"],
            f"{100 * a['success_rate']:.1f}%",
            a["cost"],
        ]
        for a in run.arms
        if a["pulls"]
    ]
    if arm_rows:
        parts += [
            "<h4>Which model and mutation the bandit used</h4>",
            table(
                ["model", "mutation", "candidates", "beat parent", "success rate", "relative cost"],
                arm_rows,
                {2, 3, 4, 5},
            ),
        ]
    if s.get("tokens"):
        calls = sum(1 for p in run.candidates if p.prompt_tokens)
        mean_s = s["llm_seconds"] / calls if calls else 0
        hit = s.get("prompt_cache_hit_rate")
        parts.append(
            f'<p class="muted">Model use: {calls} calls, {s["tokens"]:,} tokens, {mean_s:.1f} s per call on '
            f"average, prompt cache reuse {pct(100 * hit if hit is not None else None, 0)}.</p>"
        )
    return "\n".join(parts)


def cross_run_html(runs: list[RunData]) -> str:
    labels = "; ".join(f"<code>{esc(r.name)}</code>: {esc(r.model_label)}" for r in runs)
    out = [comparison_figure(runs), f'<p class="muted">{labels}.</p>']
    tested = [r for r in runs if r.sealed and "champion" in r.sealed["per_item"]]
    rows = []
    for i, a in enumerate(tested):
        for b in tested[i + 1 :]:
            pa, pb = a.sealed["per_item"]["champion"], b.sealed["per_item"]["champion"]
            keys = [k for k in pa if k in pb]
            if not keys:
                continue
            c = compare([pa[k] for k in keys], [pb[k] for k in keys])
            rows.append(
                [
                    f"{esc(a.name)} vs {esc(b.name)}",
                    f"{c['mean_diff']:+.2f} pts [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]",
                    f"{c['p_value']:.3g}",
                    f"{c['a12']:.2f} ({c['effect']})",
                    f"{c['wins']} / {c['losses']} / {c['ties']}",
                ]
            )
    if rows:
        out.append("<h4>Champions compared on the sealed test</h4>")
        out.append(
            table(
                [
                    "comparison",
                    "mean gap difference [95% CI]",
                    "p (Wilcoxon)",
                    "A12 (effect)",
                    "wins / losses / ties",
                ],
                rows,
            )
        )
    return "\n".join(out)


TASK_TITLES = {
    "cvrp_construct": "Task 1 · Constructive heuristic for vehicle routing",
    "pyvrp_accept": "Task 2 · Acceptance rule inside the PyVRP solver",
}


def plotly_script(offline: bool) -> str:
    if offline:
        from plotly.offline import get_plotlyjs

        return f'<script charset="utf-8">{get_plotlyjs()}</script>'
    return f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>'


def build_site(
    run_dirs: list[Path], out_dir: Path, title: str = "HeuriForge results", offline: bool = False
) -> Path:
    """Write ``out_dir/index.html``. ``offline`` embeds plotly.js (3.5 MB) instead of using its CDN."""
    runs = [RunData(p) for p in run_dirs if (p / "programs.sqlite").exists()]
    if not runs:
        raise FileNotFoundError("no runs found (each run folder needs a programs.sqlite)")
    demo = any(r.provider == "mock" for r in runs)
    body = [f"<h1>{esc(title)}</h1>"]
    body.append(
        '<p class="muted">LLM-guided evolution of vehicle-routing heuristics with a small local model. '
        "Gaps are relative to the best-known solutions of the instances (lower is better).</p>"
    )
    if demo:
        body.append(
            '<div class="banner"><strong>Offline demo.</strong> These candidates came from a mock model '
            "(random code mutations), to show the pipeline end to end. They are not real results.</div>"
        )
    for task in sorted({r.task for r in runs}):
        group = [r for r in runs if r.task == task]
        body.append(f"<h2>{esc(TASK_TITLES.get(task, task))}</h2>")
        if len(group) > 1:
            body.append(cross_run_html(group))
        body.extend(run_section(r) for r in group)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    body.append(
        f"<footer>Generated by HeuriForge on {stamp}. Source and method: see the repository README.</footer>"
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
{plotly_script(offline)}
<style>{CSS}</style></head>
<body><main>
{chr(10).join(body)}
</main></body></html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "index.html"
    target.write_text(page)
    (out_dir / ".nojekyll").write_text("")
    return target

"""The program database: every candidate, its lineage, scores and failure reason (SQLite).

One file per run (``runs/<name>/programs.sqlite``). Everything needed to resume a run,
draw the report or audit what the model tried is in here.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS programs (
    id INTEGER PRIMARY KEY,
    attempt INTEGER,
    parent_id INTEGER,
    island INTEGER,
    generation INTEGER,
    created_at REAL,
    code TEXT,
    code_hash TEXT,
    structure_hash TEXT,
    origin TEXT,
    model TEXT,
    mutation TEXT,
    idea TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cached_tokens INTEGER,
    llm_seconds REAL,
    status TEXT,
    error_type TEXT,
    error TEXT,
    smoke_score REAL,
    train_score REAL,
    val_score REAL,
    train_details TEXT,
    val_details TEXT,
    eval_seconds REAL,
    complexity INTEGER,
    params_used INTEGER,
    cell TEXT,
    children INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS programs_status ON programs (status);
CREATE INDEX IF NOT EXISTS programs_hash ON programs (code_hash);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY, created_at REAL, island INTEGER, text TEXT, source_ids TEXT
);
CREATE TABLE IF NOT EXISTS bandit (
    arm TEXT PRIMARY KEY, pulls INTEGER, successes INTEGER, failures INTEGER, cost REAL
);
CREATE TABLE IF NOT EXISTS archive (
    island INTEGER, cell TEXT, program_id INTEGER, PRIMARY KEY (island, cell)
);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, created_at REAL, kind TEXT, data TEXT);
"""

# Final states of a candidate, in the order a candidate moves through the pipeline.
STATUSES = [
    "llm_error",  # the model call failed or returned unusable JSON
    "edit_failed",  # the proposed edits did not apply to the code
    "rejected_static",  # unsafe or rule-breaking code (see sandbox/static.py)
    "rejected_duplicate",  # same program as one already seen, or too many constant-only variants
    "failed_smoke",  # crashed, timed out or misbehaved on the tiny smoke instances
    "failed_train",  # same, on the training instances
    "evaluated",  # has a training score
]

JSON_FIELDS = {"train_details", "val_details"}


@dataclass
class Program:
    island: int
    code: str | None
    parent_id: int | None = None
    generation: int = 0
    origin: str = "llm"  # initial | llm | random | migrated
    id: int | None = None
    attempt: int | None = None
    created_at: float = 0.0
    code_hash: str | None = None
    structure_hash: str | None = None
    model: str | None = None
    mutation: str | None = None
    idea: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    llm_seconds: float = 0.0
    status: str = "new"
    error_type: str | None = None
    error: str | None = None
    smoke_score: float | None = None
    train_score: float | None = None
    val_score: float | None = None
    train_details: dict | None = None
    val_details: dict | None = None
    eval_seconds: float = 0.0
    complexity: int = 0
    params_used: int = 0
    cell: str | None = None
    children: int = 0


_COLUMNS = [f.name for f in fields(Program)]


class Database:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path)
            self.conn.execute(
                "PRAGMA journal_mode=WAL"
            )  # `heuriforge status` can read during a run
            self.conn.executescript(SCHEMA)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    # ----------------------------------------------------------------- programs
    def _row(self, p: Program) -> dict:
        row = asdict(p)
        for key in JSON_FIELDS:
            row[key] = json.dumps(row[key]) if row[key] is not None else None
        return row

    def _program(self, row: sqlite3.Row) -> Program:
        data = {k: row[k] for k in row.keys() if k in _COLUMNS}  # noqa: SIM118 (sqlite3.Row)
        for key in JSON_FIELDS:
            if data.get(key):
                data[key] = json.loads(data[key])
        return Program(**data)

    def insert(self, p: Program) -> int:
        if not p.created_at:
            p.created_at = time.time()
        row = self._row(p)
        row.pop("id")
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        cur = self.conn.execute(f"INSERT INTO programs ({cols}) VALUES ({marks})", row)
        self.conn.commit()
        p.id = cur.lastrowid
        return p.id

    def update(self, p: Program) -> None:
        row = self._row(p)
        sets = ", ".join(f"{k} = :{k}" for k in row if k != "id")
        self.conn.execute(f"UPDATE programs SET {sets} WHERE id = :id", row)
        self.conn.commit()

    def get(self, program_id: int) -> Program | None:
        row = self.conn.execute("SELECT * FROM programs WHERE id = ?", (program_id,)).fetchone()
        return self._program(row) if row else None

    def programs(self, where: str = "1", params: tuple = ()) -> list[Program]:
        rows = self.conn.execute(f"SELECT * FROM programs WHERE {where} ORDER BY id", params)
        return [self._program(r) for r in rows]

    def evaluated(self) -> list[Program]:
        return self.programs("status = 'evaluated'")

    def attempts(self) -> int:
        """Candidates proposed so far (the budget counts these)."""
        row = self.conn.execute("SELECT MAX(attempt) FROM programs").fetchone()
        return int(row[0] or 0)

    def hash_counts(self) -> tuple[set[str], dict[str, int]]:
        """Seen code hashes, and how many programs share each structure hash."""
        seen = {
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT code_hash FROM programs WHERE code_hash IS NOT NULL"
            )
        }
        counts = {
            r[0]: r[1]
            for r in self.conn.execute(
                "SELECT structure_hash, COUNT(DISTINCT code_hash) FROM programs "
                "WHERE structure_hash IS NOT NULL AND origin != 'migrated' GROUP BY structure_hash"
            )
        }
        return seen, counts

    def add_child(self, program_id: int) -> None:
        self.conn.execute("UPDATE programs SET children = children + 1 WHERE id = ?", (program_id,))
        self.conn.commit()

    def status_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) FROM programs WHERE origin IN ('llm', 'random') GROUP BY status"
        )
        return {r[0]: r[1] for r in rows}

    def error_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT error_type, COUNT(*) FROM programs WHERE error_type IS NOT NULL GROUP BY error_type"
        )
        return {r[0]: r[1] for r in rows}

    # ----------------------------------------------------------------- meta
    def set_meta(self, key: str, value) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ----------------------------------------------------------------- lessons
    def add_lesson(self, text: str, island: int, source_ids: list[int]) -> None:
        self.conn.execute(
            "INSERT INTO lessons (created_at, island, text, source_ids) VALUES (?, ?, ?, ?)",
            (time.time(), island, text, json.dumps(source_ids)),
        )
        self.conn.commit()

    def lessons(self, limit: int | None = None) -> list[str]:
        rows = self.conn.execute("SELECT text FROM lessons ORDER BY id DESC").fetchall()
        texts = [r[0] for r in rows]
        return texts[:limit] if limit is not None else texts

    # ----------------------------------------------------------------- bandit
    def save_arm(self, arm: str, pulls: int, successes: int, failures: int, cost: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO bandit VALUES (?, ?, ?, ?, ?)",
            (arm, pulls, successes, failures, cost),
        )
        self.conn.commit()

    def arms(self) -> dict[str, dict]:
        rows = self.conn.execute("SELECT * FROM bandit ORDER BY arm")
        return {r["arm"]: dict(r) for r in rows}

    # ----------------------------------------------------------------- archive
    def save_cell(self, island: int, cell: str, program_id: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO archive VALUES (?, ?, ?)", (island, cell, program_id)
        )
        self.conn.commit()

    def cells(self) -> list[tuple[int, str, int]]:
        return [tuple(r) for r in self.conn.execute("SELECT island, cell, program_id FROM archive")]

    # ----------------------------------------------------------------- events
    def log_event(self, kind: str, data: dict) -> None:
        self.conn.execute(
            "INSERT INTO events (created_at, kind, data) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(data)),
        )
        self.conn.commit()

    def events(self, kind: str | None = None) -> list[dict]:
        if kind is None:
            rows = self.conn.execute("SELECT * FROM events ORDER BY id")
        else:
            rows = self.conn.execute("SELECT * FROM events WHERE kind = ? ORDER BY id", (kind,))
        return [
            {"created_at": r["created_at"], "kind": r["kind"], **json.loads(r["data"])}
            for r in rows
        ]

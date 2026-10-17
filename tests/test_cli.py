import pytest

from heuriforge.cli import build_parser, main


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for command in ("data", "eval", "evolve", "status", "test", "report", "bench", "demo"):
        assert command in out


def test_eval_command(project, monkeypatch, capsys):
    monkeypatch.setenv("HEURIFORGE_ROOT", str(project))
    main(["eval", "--task", "cvrp_construct", "--dataset", "tiny", "--split", "val"])
    out = capsys.readouterr().out
    assert "S-n33-s4#0" in out and "mean" in out and "savings" in out


def test_the_test_split_cannot_be_used_by_eval(project, monkeypatch):
    monkeypatch.setenv("HEURIFORGE_ROOT", str(project))
    with pytest.raises(SystemExit, match="sealed"):
        main(["eval", "--task", "cvrp_construct", "--dataset", "tiny", "--split", "test"])


def test_evolve_and_status_commands(project, monkeypatch, capsys):
    monkeypatch.setenv("HEURIFORGE_ROOT", str(project))
    main(
        [
            "evolve",
            "--task",
            "cvrp_construct",
            "--dataset",
            "tiny",
            "--name",
            "cli-random",
            "--provider",
            "random",
            "--max-candidates",
            "3",
        ]
    )
    assert (project / "runs" / "cli-random" / "log.txt").read_text().count("random/random") == 3
    main(["status", "--name", "cli-random"])
    out = capsys.readouterr().out
    assert "Candidates: 3" in out and "Champion" in out


def test_bench_against_the_mock_server(settings, capsys):
    import asyncio

    from heuriforge.bench import print_bench, run_bench
    from heuriforge.llm import mock_transport
    from heuriforge.tasks import get_task

    transport = mock_transport(get_task("cvrp_construct"), seed=0, fault_rate=0.0)
    result = asyncio.run(run_bench(settings, "cvrp_construct", 4, (1, 2), transport))
    assert [r["concurrency"] for r in result["rounds"]] == [1, 2]
    assert all(r["valid_json"] == 1.0 and r["requests"] == 4 for r in result["rounds"])
    print_bench(result)
    assert "cand/hour" in capsys.readouterr().out


def test_table_and_export(project, monkeypatch, capsys):
    import json

    monkeypatch.setenv("HEURIFORGE_ROOT", str(project))
    gaps = {"llm": {"a#0": 10.0, "b#0": 12.0}, "random": {"a#0": 11.0, "b#0": 13.0}}
    for task in ("construct", "accept"):
        for kind, per_item in gaps.items():
            run = project / "runs" / f"{task}-{kind}"
            run.mkdir(parents=True, exist_ok=True)
            report = {
                "mean_gap": {"champion": sum(per_item.values()) / 2, "initial": 20.0},
                "per_item": {"champion": per_item},
            }
            (run / "sealed_test.json").write_text(json.dumps(report))
    main(["table"])
    out = capsys.readouterr().out
    assert "| 1 · Constructive heuristic | 20.00% | 12.00% | 11.00% | -1.00 pts" in out
    main(["export", "--name", "construct-llm", "--out", str(project / "results")])
    assert (project / "results" / "construct-llm" / "sealed_test.json").exists()

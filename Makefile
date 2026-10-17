# HeuriForge: everyday commands. `make help` lists them.
# Variables you can override on the command line, for example:
#   make server MODEL=unsloth/Qwen3.5-2B-GGUF:Q4_K_M PORT=8081
#   make construct-llm RESUME=1
#   make construct-llm SEED=1        (a repeated run: folder runs/construct-llm-s1)
#   make status RUN=construct-llm

SHELL := /bin/bash
.DEFAULT_GOAL := help

MODEL ?= unsloth/Qwen3.5-4B-GGUF:Q4_K_M
PORT ?= 8080
SLOTS ?= 2
CTX ?= 16384
SERVER_FLAGS ?=
RUN ?= construct-llm
RESUME ?=
SEED ?= 0

# Keep the Mac awake during long runs (no-op on Linux).
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -is)
# Repeated runs: SEED=1, SEED=2, ... add -s1, -s2, ... to the run name (seed 0 keeps the plain name).
SUFFIX := $(if $(filter 0,$(SEED)),,-s$(SEED))
# --resume when asked (RESUME=1), or, with RESUME=auto, when the run folder already exists.
resume_flag = $(if $(filter auto,$(RESUME)),$(if $(wildcard runs/$(1)/programs.sqlite),--resume,),$(if $(RESUME),--resume,))
evolve = $(CAFFEINATE) uv run heuriforge evolve --task $(1) --name $(2)$(SUFFIX) --seed $(SEED) $(3) $(call resume_flag,$(2)$(SUFFIX))

.PHONY: help setup server bench data demo eval construct-llm construct-random accept-llm \
	accept-random repeats-llm repeats-random status sealed-test sealed-tests report export \
	export-all table test lint clean

help: ## list the commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## install Python, the dependencies and the git hooks
	@git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init -b main
	uv sync
	uv run pre-commit install

server: ## start the local model server (first time: downloads about 2.7 GB)
	llama-server -hf $(MODEL) --port $(PORT) --no-mmproj -np $(SLOTS) -c $(CTX) $(SERVER_FLAGS)

bench: ## Phase 0: measure the model server (start it first with `make server`)
	uv run heuriforge bench

data: ## download the X instances and generate the synthetic ones
	uv run heuriforge data

demo: ## offline end-to-end demo with a mock model (no server, no download)
	uv run heuriforge demo

eval: ## score both initial programs (a sanity check, about a minute)
	uv run heuriforge eval --task cvrp_construct
	uv run heuriforge eval --task pyvrp_accept

construct-llm: ## Task 1 with the language model (RESUME=1 continues, SEED=1 is a repeat)
	$(call evolve,cvrp_construct,construct-llm,)

construct-random: ## Task 1 without a language model (the baseline)
	$(call evolve,cvrp_construct,construct-random,--provider random)

accept-llm: ## Task 2 with the language model (RESUME=1 continues, SEED=1 is a repeat)
	$(call evolve,pyvrp_accept,accept-llm,)

accept-random: ## Task 2 without a language model (the baseline)
	$(call evolve,pyvrp_accept,accept-random,--provider random)

repeats-llm: ## seeds 1 and 2 of both model runs, one after the other (about 6 hours)
	$(MAKE) construct-llm SEED=1 RESUME=auto
	$(MAKE) accept-llm SEED=1 RESUME=auto
	$(MAKE) construct-llm SEED=2 RESUME=auto
	$(MAKE) accept-llm SEED=2 RESUME=auto

repeats-random: ## seeds 1 and 2 of both no-model runs (no server needed, under an hour)
	$(MAKE) construct-random SEED=1 RESUME=auto
	$(MAKE) accept-random SEED=1 RESUME=auto
	$(MAKE) construct-random SEED=2 RESUME=auto
	$(MAKE) accept-random SEED=2 RESUME=auto

status: ## progress of a run, also while it runs (RUN=construct-llm)
	uv run heuriforge status --name $(RUN)

sealed-test: ## the sealed test of a finished run, once (RUN=construct-llm)
	uv run heuriforge test --name $(RUN)

sealed-tests: ## the sealed test of every finished run that has not had one yet
	uv run heuriforge test --all

report: ## build the results website in site/
	uv run heuriforge report

export: ## copy a run's result files to results/ for committing (RUN=construct-llm)
	uv run heuriforge export --name $(RUN)

export-all: ## copy the result files of every run to results/
	uv run heuriforge export --all

table: ## print the README results table, pooled over the repeated runs
	uv run heuriforge table

test: ## run the test suite (about a minute)
	uv run pytest

lint: ## check the code style
	uv run ruff check .
	uv run ruff format --check .

clean: ## delete generated files (keeps data/ and runs/)
	rm -rf site-demo .pytest_cache .ruff_cache

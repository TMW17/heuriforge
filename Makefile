# HeuriForge: everyday commands. `make help` lists them.
# Variables you can override on the command line, for example:
#   make server MODEL=unsloth/Qwen3.5-2B-GGUF:Q4_K_M PORT=8081
#   make construct-llm RESUME=1
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

# Keep the Mac awake during long runs (no-op on Linux).
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -is)
RESUME_FLAG := $(if $(RESUME),--resume,)

.PHONY: help setup server bench data demo eval construct-llm construct-random accept-llm \
	accept-random status sealed-test report export table test lint clean

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

construct-llm: ## Task 1 with the language model (add RESUME=1 to continue)
	$(CAFFEINATE) uv run heuriforge evolve --task cvrp_construct --name construct-llm $(RESUME_FLAG)

construct-random: ## Task 1 without a language model (the baseline)
	$(CAFFEINATE) uv run heuriforge evolve --task cvrp_construct --name construct-random --provider random $(RESUME_FLAG)

accept-llm: ## Task 2 with the language model (add RESUME=1 to continue)
	$(CAFFEINATE) uv run heuriforge evolve --task pyvrp_accept --name accept-llm $(RESUME_FLAG)

accept-random: ## Task 2 without a language model (the baseline)
	$(CAFFEINATE) uv run heuriforge evolve --task pyvrp_accept --name accept-random --provider random $(RESUME_FLAG)

status: ## progress of a run, also while it runs (RUN=construct-llm)
	uv run heuriforge status --name $(RUN)

sealed-test: ## the sealed test of a finished run, once (RUN=construct-llm)
	uv run heuriforge test --name $(RUN)

report: ## build the results website in site/
	uv run heuriforge report

export: ## copy a run's result files to results/ for committing (RUN=construct-llm)
	uv run heuriforge export --name $(RUN)

table: ## print the README results table from the four sealed tests
	uv run heuriforge table

test: ## run the test suite (about a minute)
	uv run pytest

lint: ## check the code style
	uv run ruff check .
	uv run ruff format --check .

clean: ## delete generated files (keeps data/ and runs/)
	rm -rf site-demo .pytest_cache .ruff_cache

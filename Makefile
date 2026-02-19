# Navi — Cross-Project Makefile
# Run from repository root. Each target iterates over all sub-projects.

PROJECTS := contracts section-manager actor auditor
PROJECTS_DIR := projects

.PHONY: sync-all test-all lint-all format-all typecheck-all check-all clean-all help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

sync-all: ## Install dependencies in all sub-projects
	@powershell -NoProfile -Command "$$projects='$(PROJECTS)'.Split(' '); foreach ($$proj in $$projects) { Write-Host ''; Write-Host ('========== uv sync: ' + $$proj + ' =========='); Push-Location (Join-Path '$(PROJECTS_DIR)' $$proj); uv sync; Pop-Location }"

test-all: ## Run tests in all sub-projects
	@powershell -NoProfile -Command "$$projects='$(PROJECTS)'.Split(' '); foreach ($$proj in $$projects) { Write-Host ''; Write-Host ('========== pytest: ' + $$proj + ' =========='); Push-Location (Join-Path '$(PROJECTS_DIR)' $$proj); uv run pytest tests/ -ra --tb=short; Pop-Location }"

lint-all: ## Run ruff check + format check in all sub-projects
	@powershell -NoProfile -Command "$$projects='$(PROJECTS)'.Split(' '); foreach ($$proj in $$projects) { Write-Host ''; Write-Host ('========== ruff: ' + $$proj + ' =========='); Push-Location (Join-Path '$(PROJECTS_DIR)' $$proj); uv run ruff check .; uv run ruff format --check .; Pop-Location }"

format-all: ## Run ruff format in all sub-projects
	@powershell -NoProfile -Command "$$projects='$(PROJECTS)'.Split(' '); foreach ($$proj in $$projects) { Write-Host ''; Write-Host ('========== format: ' + $$proj + ' =========='); Push-Location (Join-Path '$(PROJECTS_DIR)' $$proj); uv run ruff format .; Pop-Location }"

typecheck-all: ## Run mypy strict in all sub-projects
	@powershell -NoProfile -Command "$$projects='$(PROJECTS)'.Split(' '); foreach ($$proj in $$projects) { Write-Host ''; Write-Host ('========== mypy: ' + $$proj + ' =========='); Push-Location (Join-Path '$(PROJECTS_DIR)' $$proj); uv run mypy src/; Pop-Location }"

check-all: lint-all typecheck-all test-all ## Run lint + typecheck + tests (CI)

clean-all: ## Remove .venv, __pycache__, .mypy_cache, .pytest_cache in all sub-projects
	@powershell -NoProfile -Command "$$projects='$(PROJECTS)'.Split(' '); foreach ($$proj in $$projects) { Write-Host ''; Write-Host ('========== clean: ' + $$proj + ' =========='); $$root = Join-Path '$(PROJECTS_DIR)' $$proj; Remove-Item -Recurse -Force (Join-Path $$root '.venv'), (Join-Path $$root '__pycache__'), (Join-Path $$root '.mypy_cache'), (Join-Path $$root '.pytest_cache'), (Join-Path $$root '.ruff_cache') -ErrorAction SilentlyContinue; Get-ChildItem $$root -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue }"

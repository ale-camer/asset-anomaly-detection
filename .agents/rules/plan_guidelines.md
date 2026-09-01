# Plan Guidelines

## Dependency and Environment Management
Whenever you create an implementation plan (like `docs/issue_X_plan.md`) or write instructions for a new task in this workspace, you **MUST** include a step to activate the virtual environment and install dependencies after checking out a new branch.

For example, the branch preparation step should look like this:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-X-name
source .venv/bin/activate
pip install -e ".[dev]"
```

This ensures that the local `pip` environment is synced with the latest dependencies in `pyproject.toml` and that all subsequent commands (like `pytest`, `ruff`, or `mypy`) execute properly within the `.venv`.

# Repository Guidelines

## First Step
<!-- This is here so that codex stops just trying to run things without any setup -->
- The first thing you should always do is check for the dev environment `.venv-mili-python-<pyver>`. If it exists you can source it `source .venv-mili-python-<pyver>/bin/activate` to ensure you have access to all the tools and code.
- If the dev environment does not exist, you should create it by running `source .venv.sh` and then activate it.

## Project Structure

- `src/mili/`: core library code (src-layout package).
- `src/scripts/`: packaged helper scripts; CLI entry points live here (see `pyproject.toml`).
- `tests/`: unit tests and fixtures (e.g., `tests/data/`).
- `doc/`: Sphinx documentation (`doc/source/`) and build tooling (`doc/Makefile`).

## Build, Test, and Development Commands

- Create a dev environment (LLNL workflow): `source .venv.sh` (creates `.venv-mili-python-<pyver>` and installs `.[dev]`).
- Editable install (generic): `python3 -m pip install -e '.[dev]'`
- Run tests (same as CI/tox): `python3 -m unittest discover tests`
- Run tox (all configured envs): `tox` (or a single env: `tox -e unittest` / `tox -e py313`)
- Type check: `mypy src/mili`
- Build artifacts: `python3 -m build .` (produces `dist/` wheel/sdist)
- Build docs: `cd doc && make html`

## Coding Style & Naming Conventions

- Python style: PEP 8, with **2-space indentation** (see `README.md`).
- Docstrings: Google convention (`.pydocstylerc`); keep docstrings Sphinx-friendly.
- Types: CI runs strict-ish `mypy` on `src/mili`; prefer explicit annotations for public APIs.
- Naming: modules/functions in `snake_case`, classes in `PascalCase`, constants in `UPPER_SNAKE_CASE`.

## Testing Guidelines

- Framework: `unittest` (CI runs via tox; see `tox.ini`).
- Naming: place new tests in `tests/test_*.py`; keep fixtures in `tests/data/` when needed.
- Coverage: CI enforces a minimum total coverage (see `MINIMUM_COVERAGE` in `.gitlab-ci.yml`).

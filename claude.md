# Python VSLAM

Visual SLAM (Simultaneous Localization and Mapping) implementation in Python.

## Project Overview

This is a Python application for implementing Visual SLAM algorithms. The project uses modern Python tooling with `uv` for fast, reliable dependency management.

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- `uv` package manager ([install instructions](https://github.com/astral-sh/uv))

### Initial Setup

1. Navigate to the project directory:
   ```bash
   cd python-vslam
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv sync --dev
   ```
   This creates a `.venv` directory and installs all development dependencies from `pyproject.toml`.

3. Activate the virtual environment:
   ```bash
   # On macOS/Linux:
   source .venv/bin/activate

   # On Windows:
   .venv\Scripts\activate
   ```

## Project Structure

```
python-vslam/
├── src/vslam/          # Main application code
│   └── __init__.py
├── tests/              # Test files (test_*.py or *_test.py)
├── pyproject.toml      # Project configuration and dependencies
├── claude.md           # This file
└── .claude/            # Claude Code configuration
    └── agents/         # Custom Claude agents
```

## Development Workflow

### Adding Dependencies

Add a runtime dependency:
```bash
uv add <package-name>
```

Add a development dependency (testing, linting, etc.):
```bash
uv add --dev <package-name>
```

Examples:
```bash
uv add numpy opencv-python  # Add SLAM libraries
uv add --dev black          # Add code formatter for dev
```

### Running Your Code

Run Python scripts directly:
```bash
uv run python src/vslam/main.py
```

Or activate the venv first:
```bash
source .venv/bin/activate
python src/vslam/main.py
```

### Testing

Run all tests:
```bash
uv run pytest
```

Run specific test file:
```bash
uv run pytest tests/test_feature_detection.py
```

Run with verbose output:
```bash
uv run pytest -v
```

Run with coverage:
```bash
uv add --dev pytest-cov
uv run pytest --cov=src/vslam
```

### Type Checking

Check types with mypy:
```bash
uv run mypy src/
```

### Linting and Formatting

Check code with ruff:
```bash
uv run ruff check .
```

Auto-fix issues:
```bash
uv run ruff check --fix .
```

Format code:
```bash
uv run ruff format .
```

## Common Commands Reference

| Task | Command |
|------|---------|
| Install/sync dependencies | `uv sync --dev` |
| Add runtime dependency | `uv add <package>` |
| Add dev dependency | `uv add --dev <package>` |
| Run script | `uv run python <script.py>` |
| Run tests | `uv run pytest` |
| Type check | `uv run mypy src/` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |

## Development Tips

1. **Always sync after pulling changes**: Run `uv sync --dev` to ensure you have the latest dependencies
2. **Don't edit pyproject.toml manually for deps**: Use `uv add` instead to keep lock file in sync
3. **Run tests before committing**: `uv run pytest` to catch issues early
4. **Use type hints**: Enable mypy to catch type errors during development

## Troubleshooting

### Virtual environment not found
Run `uv sync --dev` to recreate it.

### Import errors
Make sure you're running commands with `uv run` or have activated the virtual environment.

### Dependency conflicts
Try removing the lock file and resyncing:
```bash
rm uv.lock
uv sync --dev
```

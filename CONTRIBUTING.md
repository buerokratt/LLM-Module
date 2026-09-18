# Contributor Guidelines

Welcome! This guide helps you set up the project locally for development and contributions. It also explains the coding standards used in the project along with the CI checks

## Local Setup

This project uses **uv** for Python packaging and virtual environments. Python is pinned to **3.12.10** in `pyproject.toml`.


### Step 1: Clone the Repository
Clone the repository and navigate to the project directory.

```bash
cd RAG-Module
```

### Step 2: Install uv
If uv is not already installed, install it via the official installer.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart your terminal or run: source ~/.zshrc
```

### Step 3: Install the Required Python Version
Use uv to install and manage Python 3.12.10.

```bash
uv python install 3.12.10
```

### Step 4: Recreate the Virtual Environment
From the project root (where `pyproject.toml` and `uv.lock` are located), recreate the environment from the lockfile.

```bash
# Create .venv and install dependencies strictly from uv.lock
uv sync --frozen
```

**Notes:**
- `uv sync` creates a local `.venv/` directory by default and installs dependencies from `uv.lock`.
- If `--frozen` fails due to a stale lockfile, run `uv sync -p 3.12.10` (without `--frozen`) to resolve and update.

### Step 5: Activate the Environment (Optional)
Activate the virtual environment to use it directly.

```bash
source .venv/bin/activate
python -V  # Should output: Python 3.12.10
```

### Step 6: Run the Application

When running any Python programs (APIs or scripts) in a docker container or locally, always run with

```bash
uv run python app.py
```

instead of 

```bash
python3 app.py
```

This will make sure that regardless of whether you have activated the .venv environment or not uv will use virtual environment created instead of system level versions.


### Step 7: Setup pre-commit hooks

Install pre-commit hooks to ensure code quality checks run automatically before commits.

```bash
uv run pre-commit install
```


This installs git hooks that will run configured checks (linting, formatting, etc.) on staged files before each commit.

#### Validate Pre-commit Setup
Test that the hooks are working correctly:

```bash
uv run pre-commit run --all-files
```

This runs all pre-commit hooks on the entire codebase. Fix any issues that are reported.


**Note:** If pre-commit hooks fail during a commit, the commit will be blocked until you fix the issues and re-stage your changes.


For more help, check the [uv documentation](https://docs.astral.sh/uv/)

## CI Checks

### Environment check

- Located in `.github/workflows/uv-env-check.yml`

- This GitHub actions check runs to check whether there are any conflicts between the lockfile and pyproject.toml. If it fails, then there has been some dependency update to the pyproject.toml without updating the lockfile.

### Type check for Python 

- Located in `.github/workflows/pyright-type-check.yml`

- This GitHub actions checks runs the Pyright type checker across the entire code-base to check for undeclared Python variables and objects. You can check the Pyright configuration in the `pyproject.toml` file. We use a `strict` configuration, so even objects being returned through frameworks and libraries should be either type-casted or should be validated using libraries such as `Pydantic`.


### Pytest Test-cases check

- Located in `.github/workflows/pytest-testcases-check.yml`

- This GitHub actions checks runs all Pytest test-cases unders the `tests/` folder.


### Ruff Python code format check

- Located in `.github/workflows/ruff-format-check.yml`

- This GitHub actions check runs the   `ruff format --check` on the entire codebase to detect any code incompliant with the project's code formatting standards which are configured in `pyproject.toml`

#### Ruff Lint check

- Located in `.github/workflows/ruff-lint-check.yml`

- This GitHub actions check runs the `ruff check` command on the entire code base to detect any code incompliant with the project's linting standards which are configured in `pyproject.toml`

### Gitleaks check

- Located in `.github/workflows/git-leaks-check.yml`

- This GitHub actions check uses the GitLeaks open source tool to check for potential secret/key leakages in the code. There is also a pre-commit hook configured with gitleaks to detect any possible secret leaks before even committing. 


## Installing New Dependencies to the Project (Python)

If you need to add a new Python dependency, **do not run `pip install` directly**.

We use `uv` to manage environments and lockfiles so that installs are reproducible in local development, CI, and containers.

### Follow This Process:

#### 1. Add the Dependency
Use `uv add` instead of `pip install`. This ensures both `pyproject.toml` and `uv.lock` are updated together.

```bash
uv add "package-name>=x.y,<x.(y+1)"
```

- Use a bounded version range (`>=` + `<`) to avoid uncontrolled upgrades.


#### 2. Re-sync Your Environment
After adding, re-sync to refresh your local `.venv`:

```bash
uv sync --reinstall
```

#### 3. Run Checks Locally
Make sure type checks, linter, and tests pass:

```bash
uv run pyright
uv run ruff check .
uv run pytest
```

#### 4. Commit Both Files
Always commit both `pyproject.toml` and `uv.lock`. If only one is updated, CI will fail (`uv sync --frozen` check).

```bash
git add pyproject.toml uv.lock
git commit -m "added package-name dependency"
```

#### 5. Open a PR
CI will validate that the lockfile and environment are consistent. If you forgot to update the lockfile, the PR will fail with a clear error.

---

## Type Safety Practices

Python is a dynamically typed language. This flexibility makes Python productive and expressive, but it also increases the risk of subtle bugs caused by incorrect function calls, unexpected None values, or inconsistent data structures.To balance flexibility with long-term maintainability we use [Pyright](https://microsoft.github.io/pyright) for CI level type-checking. 

We run Pyright in `standard` mode. This mode provides strong type correctness guarantees without requiring the full strictness and annotation overhead of `strict` mode.

You can check the exact type checking constraints enforced in `standard` mode here in the `Diagnostic Defaults` section of the [Pyright documentation](https://microsoft.github.io/pyright/#/configuration?id=diagnostic-settings-defaults).

`standard` mode in Pyright is chosen because it enforces the following principles: 

- **Catch real bugs early** - It prevents incorrect function calls, invalid attribute access, misuse of Optional values, inconsistent overloads, and a wide range of type errors that would otherwise only appear at runtime.

- **Maintain clarity without excessive annotation burden** - Developers are not expected to annotate every variable or build fully typed signatures for every function. Pyright uses inference aggressively, and `standard` mode focuses on correctness where types are known or inferred.

- **Work seamlessly with third-party libraries** - Many Python libraries ship without type stubs. In `standard` mode, these imports are treated as Any, allowing us to use them without blocking type checks while still preserving type safety inside our own code. 

### Runtime Type Safety at System Boundaries

While Pyright provides excellent static type checking during development, **system boundaries** require additional runtime validation. These are points where our Python code interfaces with external systems, user input, or network requests where data types cannot be guaranteed at compile time.

In this project, we use **Pydantic** for rigorous runtime type checking at these critical handover points:

#### FastAPI Endpoints
All FastAPI route handlers use Pydantic models for request/response validation:
- Request bodies are validated against Pydantic schemas
- Query parameters and path parameters are type-checked at runtime
- Response models ensure consistent API contract enforcement
```python
# Example: API endpoint with Pydantic validation
from pydantic import BaseModel
from fastapi import FastAPI


class UserRequest(BaseModel):
    name: str
    age: int


@app.post("/users")
async def create_user(user: UserRequest):
    # Pydantic validates name is string, age is int
    # Invalid data raises 422 before reaching this code
    return {"id": 1, "name": user.name}
```

This dual approach of **static type checking with Pyright** + **runtime validation with Pydantic** ensures both development-time correctness and production-time reliability at system boundaries where type safety cannot be statically guaranteed.

**Note: Type checks are only run on core source code and not on test-cases**

## Linter Rules

Consistent linting is essential for maintaining a reliable and scalable code-base. By adhering to a well-defined linter configuration, we ensure the code remains readable, secure, and predictable even as the project evolves.

The following set of rules are enabled in this repository. Linter rules are enforced automatically through the CI pipeline and must pass before merging changes into the `wip`, `dev`, or `main` branches.
. 

Each category is summarized with a description and a link to the Ruff documentation explaining these rules.

### Selected Linter Rule Categories

#### E4, E7, E9 — Pycodestyle Error Rules

These check for fundamental correctness issues such as import formatting, indentation, and syntax problems that would otherwise cause runtime failures.

- **E4**: Import formatting and blank-line rules  
    (https://docs.astral.sh/ruff/rules/#pycodestyle-e4)

- **E7**: Indentation and tab-related issues  
(https://docs.astral.sh/ruff/rules/#pycodestyle-e7)

- **E9**: Syntax errors and runtime error patterns (e.g., undefined names in certain contexts)  
(https://docs.astral.sh/ruff/rules/#pycodestyle-e9)

#### F — Pyflakes

Static analysis rules that detect real bug patterns such as unused variables, unused imports, undefined names, duplicate definitions, and logical mistakes that can cause bugs.

(https://docs.astral.sh/ruff/rules/#pyflakes-f)

#### B — Flake8-Bugbear

A set of high-value checks for common Python pitfalls: mutable default arguments, improper exception handling, unsafe patterns, redundant checks, and subtle bugs that impact correctness and security.

(https://docs.astral.sh/ruff/rules/#flake8-bugbear-b)

#### T20 — Flake8-Print

Flags any usage of `print()` or `pprint()` in production code to prevent leaking sensitive information, mixing debug output into logs, or introducing uncontrolled console output.

(https://docs.astral.sh/ruff/rules/#flake8-print-t20)

#### N — PEP8-Naming

Ensures consistent and conventional naming across classes, functions, variables, and modules. This helps maintain readability across the engineering team and reinforces clarity in code reviews.

(https://docs.astral.sh/ruff/rules/#pep8-naming-n)

#### ANN — Flake8-Annotations

Enforces type annotation discipline across functions, methods, and class structures. With Pyright used for type checking, these rules ensure that type information remains explicit and complete.

(https://docs.astral.sh/ruff/rules/#flake8-annotations-ann)

#### ERA — Eradicate

Removes or flags commented-out code fragments. Commented code tends to accumulate over time and reduces clarity. The goal is to keep the repository clean and avoid keeping dead code in version control.

(https://docs.astral.sh/ruff/rules/#eradicate-era)

#### PERF — Perflint

Performance-oriented rules that highlight inefficient constructs, slow loops, unnecessary list or dict operations, and patterns that degrade runtime efficiency.

(https://docs.astral.sh/ruff/rules/#perflint-perf)

### Fixing Linting Issues

Linting issues should always be resolved manually. 
We **strongly discourage** relying on autofixes using `ruff check --fix` for this repository.

Unlike `ruff format`, which performs safe and predictable code formatting, the linter's autofix mode can alter control flow, refactor logic, or rewrite expressions in ways that introduce unintended bugs.

All linter errors will have **rule-code** like `ANN204` for example. 
You can use the command line command
```bash
ruff rule <rule-code> #for example: ANN204
```

to get an explanation on the rule code, why it's a problem and how you can fix it. 

Human oversight is essential to ensure that any corrective changes maintain the intended behavior of the application. Contributors should review each reported linting issue, understand why it is flagged, and apply the appropriate fix by hand.

---

## Formatting Rules

This repository uses the **Ruff Formatter** for code formatting. Its behavior is deterministic, safe, and aligned with the [Black Code Style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html).

Formatting is enforced automatically through the CI pipeline and must pass before merging changes into the `wip`, `dev`, or `main` branches.

### Selected Formatting Behaviors

#### String Quote Style

All string literals are formatted using **double quotes**.
This preserves consistency across the codebase and avoids unnecessary formatting churn.

(https://docs.astral.sh/ruff/formatter/#quote-style)

#### Indentation Style

Indentation always uses **spaces, not tabs**.
This mirrors the formatting style adopted by Black and avoids ambiguity across editors and environments.

(https://docs.astral.sh/ruff/formatter/#indent-style)

#### Magic Trailing Commas

The formatter respects magic trailing commas, meaning:

- **Adding a trailing comma** in lists, dicts, tuples, or function calls will trigger multi-line formatting.
- **Removing a trailing comma** results in a more compact single-line layout where appropriate.

This produces stable diffs and predictable wrapping behavior.

(https://docs.astral.sh/ruff/formatter/#skip-magic-trailing-comma)

#### Automatic Line Ending Detection

Ruff automatically detects and preserves the correct line-ending style (LF or CRLF) based on the existing file.
This prevents accidental line-ending changes when multiple developers work on different systems.

(https://docs.astral.sh/ruff/formatter/#line-ending)

#### Docstring Code Blocks

The formatter **does not reformat** code blocks inside docstrings.
This ensures that examples, snippets, API usage patterns, and documentation content remain exactly as written, preventing unintended modifications to teaching material or markdown-style fenced blocks.

(https://docs.astral.sh/ruff/formatter/#docstring-code-format)

### Applying Formatting

Unlike lint autofixes, **formatting changes are safe by design**.
The formatter never changes logical behavior, control flow, or semantics. It only standardizes layout.

You can run formatting locally using:

```bash
uv run ruff format
```

All formatting issues must be resolved before creating a pull request or merging into protected branches.



---

### Important Notes

- **Never edit `uv.lock` manually.** It is controlled by `uv`.
- **Never use `uv pip install` for permanent deps** — it only changes your local venv. Use `uv add` instead.
- **Never add or depend on `requirement.txt` files** for installing packages locally or through docker containers. Use `uv run sync --frozen` instead.


- If you remove a dependency, run:

```bash
uv remove package-name
uv sync --reinstall
git add pyproject.toml uv.lock
git commit -m "removed package-name"
```

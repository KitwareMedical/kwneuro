"""Convert Jupytext notebooks to executed MyST Markdown pages for Sphinx docs.

Usage:
    # Convert all notebooks (needs all extras installed)
    uv run --extra notebooks --extra all python scripts/update-notebook-pages.py

    # Convert specific notebooks
    uv run --extra notebooks --extra combat python scripts/update-notebook-pages.py \
        notebooks/example-harmonization.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
TUTORIALS_DIR = REPO_ROOT / "docs" / "tutorials"


def check_dependencies() -> None:
    """Verify that jupytext and nbconvert are available."""
    for tool in ["jupytext", "jupyter-nbconvert"]:
        if shutil.which(tool) is None:
            print(
                f"Error: '{tool}' not found. Install with: uv sync --extra notebooks",
                file=sys.stderr,
            )
            sys.exit(1)


def convert_notebook(py_path: Path) -> bool:
    """Convert a single Jupytext .py notebook to executed Markdown.

    Returns True on success, False on failure.
    """
    name = py_path.stem
    output_md = TUTORIALS_DIR / f"{name}.md"
    output_files_dir = TUTORIALS_DIR / f"{name}_files"

    print(f"\n{'=' * 60}")
    print(f"Converting: {py_path.name}")
    print(f"{'=' * 60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        ipynb_path = tmpdir_path / f"{name}.ipynb"

        # Step 1: Jupytext .py -> .ipynb
        print("  [1/3] Converting to ipynb...")
        result = subprocess.run(
            ["jupytext", "--to", "ipynb", "--output", str(ipynb_path), str(py_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED (jupytext): {result.stderr}", file=sys.stderr)
            return False

        # Step 2: Execute the notebook
        print("  [2/3] Executing notebook...")
        result = subprocess.run(
            [
                "jupyter-nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                str(ipynb_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED (execute): {result.stderr}", file=sys.stderr)
            return False

        # Step 3: Convert executed notebook to Markdown
        print("  [3/3] Exporting to Markdown...")
        result = subprocess.run(
            [
                "jupyter-nbconvert",
                "--to",
                "markdown",
                "--output-dir",
                str(tmpdir_path),
                str(ipynb_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED (markdown export): {result.stderr}", file=sys.stderr)
            return False

        # Step 4: Move outputs to docs/tutorials/
        tmp_md = tmpdir_path / f"{name}.md"
        tmp_files = tmpdir_path / f"{name}_files"

        if not tmp_md.exists():
            print(f"  FAILED: expected {tmp_md} not found", file=sys.stderr)
            return False

        # Clean previous output
        if output_md.exists():
            output_md.unlink()
        if output_files_dir.exists():
            shutil.rmtree(output_files_dir)

        shutil.copy2(tmp_md, output_md)
        if tmp_files.exists():
            shutil.copytree(tmp_files, output_files_dir)

    print(f"  OK -> {output_md.relative_to(REPO_ROOT)}")
    if output_files_dir.exists():
        n_images = len(list(output_files_dir.glob("*.png")))
        print(f"       + {n_images} image(s) in {output_files_dir.name}/")
    return True


def main() -> None:
    check_dependencies()
    TUTORIALS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which notebooks to convert
    if len(sys.argv) > 1:
        notebooks = [Path(arg).resolve() for arg in sys.argv[1:]]
        for nb in notebooks:
            if not nb.exists():
                print(f"Error: {nb} not found", file=sys.stderr)
                sys.exit(1)
            if nb.suffix != ".py":
                print(f"Error: {nb} is not a .py file", file=sys.stderr)
                sys.exit(1)
    else:
        notebooks = sorted(NOTEBOOKS_DIR.glob("example-*.py"))
        if not notebooks:
            print("No example-*.py notebooks found in notebooks/", file=sys.stderr)
            sys.exit(1)

    print(f"Will convert {len(notebooks)} notebook(s):")
    for nb in notebooks:
        print(f"  - {nb.relative_to(REPO_ROOT)}")

    # Convert each notebook
    succeeded = []
    failed = []
    for nb in notebooks:
        if convert_notebook(nb):
            succeeded.append(nb.stem)
        else:
            failed.append(nb.stem)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
    if succeeded:
        print(f"  Succeeded: {', '.join(succeeded)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\nOutput directory: {TUTORIALS_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

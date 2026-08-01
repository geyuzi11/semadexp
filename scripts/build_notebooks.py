"""Build narrative .ipynb notebooks from `notebooks/src/*.py` sources.

Cell convention:
  # %% [markdown]   -> markdown cell (following lines starting with '# ' are the text)
  # %%              -> code cell
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def build(source: Path, out_dir: Path) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines()
    cells: list[dict] = []
    cur_type: str | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_type, cur_lines
        if cur_type is None:
            return
        if cur_type == "markdown":
            text = "\n".join(
                line[2:] if line.startswith("# ") else line
                for line in cur_lines
            ).strip()
            if text:
                cells.append({"cell_type": "markdown", "metadata": {}, "source": text + "\n"})
        else:
            code = "\n".join(cur_lines).strip()
            if code:
                cells.append(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": code + "\n",
                    }
                )
        cur_lines = []

    for line in lines:
        m = re.match(r"# %%\s*(?:\[(markdown)\])?", line)
        if m:
            flush()
            cur_type = "markdown" if m.group(1) else "code"
        elif cur_type is not None:
            cur_lines.append(line)
    flush()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = out_dir / (source.stem + ".ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src_dir = root / "notebooks" / "src"
    out_dir = root / "notebooks"
    for source in sorted(src_dir.glob("*.py")):
        print(f"built {build(source, out_dir)}")


if __name__ == "__main__":
    main()


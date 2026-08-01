#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

echo "== 1/3 运行测试套件 =="
"$PY" -m pytest -q

echo "== 2/3 构建 Notebooks =="
"$PY" scripts/build_notebooks.py

echo "== 3/3 快速全流程冒烟（quick 规模）=="
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/semadexp_mplcache}" "$PY" -m semadexp.cli run-all --scale quick --out results_quick

echo "== 验证完成 =="

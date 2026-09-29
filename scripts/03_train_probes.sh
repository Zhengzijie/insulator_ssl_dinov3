#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/experiment.yaml}"
python src/run_stage.py probes --config "$config"


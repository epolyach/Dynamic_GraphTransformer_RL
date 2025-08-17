#!/usr/bin/env bash
set -euo pipefail

# Execute all prepared runs in numeric order
for d in $(ls -1d i_* | sort); do
  echo "Running $d"
  (cd "$d" && ./run.sh)
done

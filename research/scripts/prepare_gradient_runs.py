#!/usr/bin/env python3
"""
Prepare per-iteration gradient-search runs from a tracking CSV and a default YAML.
- Reads results/smart_search/param_tracking.csv
- Reads configs/gat_rl_gradient_defaults.yaml
- For each CSV row, creates results/gradient_search/i_XXX_<short>/
- Writes derived YAML with overrides
- Writes run.sh with a placeholder command (edit to your trainer entry point)
"""
import csv
import os
import sys
from pathlib import Path

try:
    import yaml  # PyYAML
except Exception as e:
    print("ERROR: PyYAML is required (pip install pyyaml)")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results/smart_search/param_tracking.csv"
DEFAULT_YAML = ROOT / "configs/gat_rl_gradient_defaults.yaml"
OUT_ROOT = ROOT / "results/gradient_search"

PARAM_KEYS = [
    ("training.learning_rate", "learning_rate", float),
    ("model.hidden_dim", "embed_dim", int),
    ("model.num_layers", "n_layers_enc", int),
    ("model.num_heads", "n_heads", int),
    ("model.transformer_dropout", "dropout", float),
    ("model.attn_dropout", "attn_dropout", float),
]


def _set_nested(d, dotted_key, value):
    keys = dotted_key.split('.')
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _short_name(row):
    # Create a compact run folder suffix from parameters
    return (
        f"lr{float(row['learning_rate']):.6g}_"
        f"E{int(float(row['embed_dim']))}_"
        f"L{int(float(row['n_layers_enc']))}_"
        f"H{int(float(row['n_heads']))}_"
        f"do{float(row['dropout']):.2f}_"
        f"ado{float(row['attn_dropout']):.2f}"
    )


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)
    if not DEFAULT_YAML.exists():
        print(f"Default YAML not found: {DEFAULT_YAML}")
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with open(DEFAULT_YAML, 'r') as f:
        base_cfg = yaml.safe_load(f)

    runs = []
    with open(CSV_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expect columns: i,model,learning_rate,embed_dim,n_layers_enc,n_heads,dropout,attn_dropout
            i = row.get('i')
            model = row.get('model', 'GAT+RL')
            if i is None:
                continue
            # derive folder
            suffix = _short_name(row)
            run_dir = OUT_ROOT / f"i_{int(i):03d}_{suffix}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # merge config
            cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy via serialize/parse
            # set model architecture from model string (kept as GAT here)
            for dotted, csv_key, caster in PARAM_KEYS:
                val_str = row[csv_key]
                val = caster(float(val_str)) if caster is int else caster(val_str)
                _set_nested(cfg, dotted, val)

            # write derived yaml
            derived_yaml = run_dir / "config.yaml"
            with open(derived_yaml, 'w') as yf:
                yaml.safe_dump(cfg, yf, sort_keys=False)

            # write run.sh with placeholder trainer command
            run_sh = run_dir / "run.sh"
            entry_py = ROOT / "run_train_validation.py"  # adjust here if you prefer a different entry
            repo_root = str(ROOT)
            cmd = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "# Auto-generated. Edit the line(s) below to switch working directory or entry point if needed.\n"
                f"REPO_ROOT=\"{repo_root}\"\n"
                "pushd \"$REPO_ROOT\" >/dev/null\n"
                f"python {entry_py} --config {derived_yaml}\n"
                "popd >/dev/null\n"
            )
            with open(run_sh, 'w') as rf:
                rf.write(cmd)
            os.chmod(run_sh, 0o755)

            runs.append(run_dir)

    # write top-level run_all.sh
    run_all = OUT_ROOT / "run_all.sh"
    with open(run_all, 'w') as raf:
        raf.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        raf.write("# Execute all prepared runs in numeric order\n")
        raf.write("for d in $(ls -1d i_* | sort); do\n")
        raf.write("  echo \"Running $d\"\n")
        raf.write("  (cd \"$d\" && ./run.sh)\n")
        raf.write("done\n")
    os.chmod(run_all, 0o755)

    print(f"Prepared {len(runs)} runs under {OUT_ROOT}")
    print("Edit run.sh in each folder to point to your desired training entry point if needed.")


if __name__ == "__main__":
    main()


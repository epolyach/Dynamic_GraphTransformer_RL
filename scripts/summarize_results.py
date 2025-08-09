#!/usr/bin/env python3
import os, csv

def main():
    base = "results_small"
    rows = []
    print(f"[INFO] Starting scan in base directory: {base}", flush=True)

    pipelines = ["cpu", "gpu", "gpu_amp"]
    models = ["pointer_rl", "static_rl", "dynamic_gt_rl"]

    total = len(pipelines) * len(models)
    count = 0
    for pipeline in pipelines:
        print(f"[INFO] Processing pipeline: {pipeline}", flush=True)
        for model in models:
            count += 1
            print(f"[INFO]  ({count}/{total}) Model: {model}", flush=True)
            d = f"{base}/{pipeline}_{model}_C20_I100_E10_B4"
            hist = os.path.join(d, "train_history.csv")
            best_png = os.path.join(d, "best_route.png")
            best_json = os.path.join(d, "best_route.json")
            print(f"[DEBUG]   Checking directory: {d}", flush=True)
            if not os.path.isdir(d):
                print(f"[WARN ]   Directory missing: {d}", flush=True)
            if not os.path.isfile(hist):
                print(f"[WARN ]   History file missing: {hist}", flush=True)
                rows.append((pipeline, model, "MISSING", "", "", ""))
                continue
            try:
                print(f"[INFO]   Reading history: {hist}", flush=True)
                with open(hist, newline="") as f:
                    r = list(csv.DictReader(f))
                print(f"[DEBUG]   Rows read: {len(r)}", flush=True)
                final = r[-1] if r else None
                val = final.get("val_cost_per_customer") if final else ""
                t = final.get("train_time_s") if final else ""
                png_ok = os.path.isfile(best_png)
                json_ok = os.path.isfile(best_json)
                print(f"[DEBUG]   best_route.png exists: {png_ok}", flush=True)
                print(f"[DEBUG]   best_route.json exists: {json_ok}", flush=True)
                rows.append((pipeline, model, val, t, best_png if png_ok else "", best_json if json_ok else ""))
                print(f"[INFO]   Appended row for {pipeline}/{model}: val={val}, train_time={t}", flush=True)
            except Exception as e:
                print(f"[ERROR]   Failed processing {hist}: {e}", flush=True)
                rows.append((pipeline, model, "ERROR", "", "", ""))

    print(f"[INFO] Finished scanning. Total rows: {len(rows)}", flush=True)

    # print a markdown-like table
    cols = ["pipeline", "model", "final_val_cost_per_customer", "last_epoch_train_time_s", "best_route.png", "best_route.json"]
    all_rows = [cols] + rows if rows else [cols]
    w = [max(len(str(x[i])) for x in all_rows) for i in range(len(cols))]
    fmt = lambda r: " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r))
    print(fmt(cols), flush=True)
    print("-|-".join("-"*wi for wi in w), flush=True)
    for r in rows:
        print(fmt(r), flush=True)

if __name__ == "__main__":
    main()

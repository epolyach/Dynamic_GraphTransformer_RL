#!/usr/bin/env python3
"""
Extra runner for OR-Tools GLS benchmarks with per-configuration threads.

Usage example:
  # Run with per-config threads (N:timeout=threads)
  # N in {50,100}; timeout in seconds
  # This schedules all requested configs concurrently with given threads.
  ./scripts/run_ortools_gls_extras.py \
    --threads 50:10=2 --threads 50:20=4 \
    --threads 100:10=2 --threads 100:20=4 --threads 100:30=6

This script uses a global ThreadPoolExecutor whose max workers equals the
sum of all per-config thread counts, and internally dispatches batches of
100 instances per task to the OR-Tools benchmark script, with robust
checkpointing per configuration so it can resume after interruptions.

Outputs will be placed under:
  results/ortools_gls_{timeout}s_{mode}/  where mode in {test, production}
with per-config checkpoints named:
  checkpoint_n{N}_timeout{timeout}.json

Requires:
  - Virtualenv at ../venv (relative to benchmark_gpu/)
  - scripts/benchmark_ortools_gls_fixed.py
"""

import argparse
import os
import sys
import time
import json
import signal
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Safe capacity mapping for supported N
CAPACITY_BY_N = {50: 40, 100: 50}

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n[{datetime.now()}] Shutdown requested. Finishing current batches...")
    shutdown_requested = True


def parse_threads_args(args_list):
    """Parse repeated --threads entries of the form N:timeout=threads"""
    configs = []
    for entry in args_list or []:
        try:
            left, threads_str = entry.split('=')
            n_str, timeout_str = left.split(':')
            n = int(n_str)
            timeout = int(timeout_str)
            threads = int(threads_str)
            if n not in CAPACITY_BY_N:
                raise ValueError(f"Unsupported N={n}; supported: {sorted(CAPACITY_BY_N.keys())}")
            if threads <= 0:
                raise ValueError("threads must be positive")
            configs.append((n, timeout, threads))
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid --threads entry '{entry}': {e}")
    return configs


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(checkpoint_path, data):
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, checkpoint_path)


def run_batch_task(n, cap, timeout, batch_size, start_idx, output_dir, venv_python):
    """Run a single batch by invoking the benchmark script for 'batch_size' instances.
    Returns (success: bool, error_msg: Optional[str]).
    """
    cmd = [
        venv_python, 'scripts/benchmark_ortools_gls_fixed.py',
        '--n', str(n),
        '--capacity', str(cap),
        '--instances', str(batch_size),
        '--timeout', str(timeout)
    ]

    try:
        # Rough timeout: per-instance time + small overhead buffer
        per_batch_timeout = batch_size * timeout + 120
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=per_batch_timeout)
        if res.returncode == 0:
            # Move any produced JSON files into output_dir with batch prefix
            os.makedirs(output_dir, exist_ok=True)
            for jf in os.listdir('.'):
                if jf.startswith(f'ortools_gls_n{n}_') and jf.endswith('.json'):
                    new_name = f"batch_{start_idx:05d}_{jf}"
                    os.replace(jf, os.path.join(output_dir, new_name))
            return True, None
        else:
            return False, res.stderr.strip()[:500]
    except subprocess.TimeoutExpired:
        return False, "batch process timeout"
    except Exception as e:
        return False, str(e)[:500]


def main():
    parser = argparse.ArgumentParser(description='Run extra OR-Tools GLS configs with per-config threads')
    parser.add_argument('--threads', action='append', default=[],
                        help='Per-config threads in form N:timeout=threads, e.g., 50:10=2 (repeatable)')
    parser.add_argument('--instances', type=int, default=10000, help='Instances per configuration (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=100, help='Instances per batch (default: 100)')
    parser.add_argument('--venv', default='../venv/bin/python3', help='Path to Python in venv')
    parser.add_argument('--mode', choices=['test', 'production'], default='test',
                        help='Output mode: test or production (default: test)')

    args = parser.parse_args()

    # Parse configs
    parsed = parse_threads_args(args.threads)
    if not parsed:
        print("No --threads entries provided. Nothing to run.")
        return 0

    # Build configs with paths and bookkeeping
    configs = []
    for n, timeout, threads in parsed:
        cap = CAPACITY_BY_N[n]
        out_dir = f'results/ortools_gls_{timeout}s_{args.mode}'
        checkpoint = os.path.join(out_dir, f'checkpoint_n{n}_timeout{timeout}.json')
        configs.append({
            'n': n,
            'cap': cap,
            'timeout': timeout,
            'threads': threads,
            'output_dir': out_dir,
            'checkpoint': checkpoint,
            'instances': args.instances,
            'batch_size': args.batch_size,
        })

    total_threads = sum(c['threads'] for c in configs)

    # Signals for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 70)
    print("OR-Tools GLS Extras Runner (per-config threads)")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Total threads: {total_threads}")
    print("Configurations:")
    for c in configs:
        print(f"  - N={c['n']:3d}, Cap={c['cap']:2d}, Timeout={c['timeout']:2d}s, Threads={c['threads']}")
    print()

    # Prepare per-config queues from checkpoints
    for c in configs:
        os.makedirs(c['output_dir'], exist_ok=True)
        ck = load_checkpoint(c['checkpoint'])
        num_batches = (c['instances'] + c['batch_size'] - 1) // c['batch_size']
        completed = set(ck.get('completed_batches', []))
        failed = set(ck.get('failed_batches', []))
        c['num_batches'] = num_batches
        c['completed'] = completed
        c['failed'] = failed
        c['in_flight'] = 0
        c['next_batches'] = [i for i in range(num_batches) if i not in completed and i not in failed]
        print(f"Config N={c['n']}, t={c['timeout']}s: {len(completed)*c['batch_size']}/{c['instances']} done; remaining batches={len(c['next_batches'])}")

    # Early exit if nothing to do
    if all(len(c['next_batches']) == 0 for c in configs):
        print("Nothing to process. All configs completed.")
        return 0

    # Global executor for all configs
    futures = {}
    with ThreadPoolExecutor(max_workers=total_threads) as executor:
        def maybe_schedule_tasks():
            scheduled_any = False
            for c in configs:
                while c['in_flight'] < c['threads'] and c['next_batches'] and not shutdown_requested:
                    batch_idx = c['next_batches'].pop(0)
                    start_idx = batch_idx * c['batch_size']
                    actual_batch_size = min(c['batch_size'], c['instances'] - start_idx)
                    c['in_flight'] += 1
                    fut = executor.submit(
                        run_batch_task,
                        c['n'], c['cap'], c['timeout'], actual_batch_size, start_idx, c['output_dir'], os.path.abspath(args.venv)
                    )
                    futures[fut] = (c, batch_idx)
                    scheduled_any = True
            return scheduled_any

        # Initial scheduling
        maybe_schedule_tasks()

        # Process completions and keep scheduling
        while futures:
            try:
                completed_any = False
                for fut in as_completed(list(futures.keys()), timeout=5):
                    completed_any = True
                    c, batch_idx = futures.pop(fut)
                c['in_flight'] -= 1
                ok, err = fut.result()
                if ok:
                    c['completed'].add(batch_idx)
                    print(f"[{datetime.now()}] N={c['n']}, t={c['timeout']}s, batch {batch_idx+1}/{c['num_batches']} ✓")
                else:
                    c['failed'].add(batch_idx)
                    print(f"[{datetime.now()}] N={c['n']}, t={c['timeout']}s, batch {batch_idx+1}/{c['num_batches']} ✗ ({err})")
                    # simple one retry policy: requeue once if not shutdown
                    if not shutdown_requested:
                        c['next_batches'].insert(0, batch_idx)

                # Update checkpoint for this config
                ck = {
                    'completed_batches': sorted(list(c['completed'])),
                    'failed_batches': sorted(list(c['failed'])),
                    'last_update': datetime.now().isoformat(),
                    'instances': c['instances'],
                    'batch_size': c['batch_size'],
                    'timeout': c['timeout'],
                    'n': c['n'],
                }
                save_checkpoint(c['checkpoint'], ck)

                # Schedule more if possible
                maybe_schedule_tasks()

                # If nothing completed in this window, just try scheduling again
                if not completed_any:
                    maybe_schedule_tasks()
            except TimeoutError:
                # No futures completed within timeout; keep waiting and scheduling
                maybe_schedule_tasks()

    # Summary
    print()
    print("=" * 70)
    print("Extras Runner Summary")
    print("=" * 70)
    for c in configs:
        done_instances = len(c['completed']) * c['batch_size']
        status = "Complete" if done_instances >= c['instances'] else f"{done_instances}/{c['instances']}"
        print(f"N={c['n']:3d}, t={c['timeout']:2d}s, threads={c['threads']}: {status}")

    print(f"\nEnd time: {datetime.now()}")
    return 0

if __name__ == '__main__':
    sys.exit(main())

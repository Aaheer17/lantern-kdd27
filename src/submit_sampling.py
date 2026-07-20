#!/usr/bin/env python3
"""submit_sampling.py — submit sample-generation jobs for every run in the
audit CSV.

For each row of runs_summary.csv (produced by audit_runs.py), submits:
    sbatch --job-name=... <job_script> <results_dir>/<folder>
where <job_script> is your existing SLURM file that runs
    python3 main.py ${RUN_DIR}/params.yaml --use_cuda --plot --generate -d ${RUN_DIR}
(remove '-ep gen' from it so results land in eval/ and samples_.hdf5).

Readiness checks per folder (skipped with a printed reason otherwise):
  * params.yaml exists
  * model.pt exists
  * no un-archived samples_.hdf5 (would be OVERWRITTEN by the new job);
    archive first with audit_runs.py --apply, or pass --force

Dry-run by default: prints every sbatch command without submitting.

Examples:
  python submit_sampling.py --runs-csv runs_summary.csv \
      --results-dir ./results --job-script sample_job.slurm
  python submit_sampling.py ... --submit
  python submit_sampling.py ... --where loss_type=voxel --submit
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-csv", required=True,
                   help="CSV produced by audit_runs.py")
    p.add_argument("--results-dir", required=True,
                   help="Directory that directly contains the run folders")
    p.add_argument("--job-script", required=True,
                   help="SLURM job script taking RUN_DIR as its argument")
    p.add_argument("--submit", action="store_true",
                   help="Actually submit (default: dry-run, print only)")
    p.add_argument("--force", action="store_true",
                   help="Submit even if an un-archived samples_.hdf5 exists "
                        "(the job will OVERWRITE it)")
    p.add_argument("--where", action="append", default=[],
                   help="Filter rows, e.g. --where loss_type=voxel. "
                        "Repeatable; conditions are ANDed.")
    args = p.parse_args()

    job_script = Path(args.job_script)
    if not job_script.is_file():
        sys.exit(f"Job script not found: {job_script}")

    df = pd.read_csv(args.runs_csv)
    if "folder" not in df.columns:
        sys.exit(f"{args.runs_csv} has no 'folder' column -- is this the "
                 "audit_runs.py output?")

    for cond in args.where:
        if "=" not in cond:
            sys.exit(f"--where expects column=value, got: {cond}")
        col, val = cond.split("=", 1)
        if col not in df.columns:
            sys.exit(f"--where column '{col}' not in {list(df.columns)}")
        df = df[df[col].astype(str) == val]
    df = df.reset_index(drop=True)
    if args.where:
        print(f"Filter {args.where}: {len(df)} rows remain")

    n_submitted = n_skipped = 0
    for _, row in df.iterrows():
        folder = str(row["folder"])
        run_dir = Path(args.results_dir) / folder

        # ---- readiness checks ----
        reasons = []
        if not run_dir.is_dir():
            reasons.append("folder missing")
        else:
            if not (run_dir / "params.yaml").is_file():
                reasons.append("no params.yaml")
            if not (run_dir / "model.pt").is_file():
                reasons.append("no model.pt")
            if (run_dir / "samples_.hdf5").exists() and not args.force:
                reasons.append("un-archived samples_.hdf5 (run "
                               "audit_runs.py --apply first, or --force)")
        if reasons:
            print(f"SKIP {folder}: " + "; ".join(reasons))
            n_skipped += 1
            continue

        job_name = f"sample_{folder}"
        cmd = ["sbatch",
               f"--job-name={job_name}",
               f"--output=logs/{job_name}_%j.out",
               f"--error=logs/{job_name}_%j.err",
               str(job_script), str(run_dir)]

        if args.submit:
            Path("logs").mkdir(exist_ok=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"SUBMITTED {folder}: {r.stdout.strip()}")
                n_submitted += 1
            else:
                print(f"FAILED {folder}: {r.stderr.strip()}")
                n_skipped += 1
        else:
            print("WOULD RUN: " + " ".join(cmd))
            n_submitted += 1

    verb = "submitted" if args.submit else "would submit"
    print(f"\n{verb}: {n_submitted} | skipped: {n_skipped} "
          f"| total rows: {len(df)}")
    if not args.submit:
        print("Dry-run only. Pass --submit to actually submit.")


if __name__ == "__main__":
    main()
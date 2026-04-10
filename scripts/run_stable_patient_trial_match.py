#!/usr/bin/env python3
"""
Run match_patients_trials_qwen.py against an OpenAI-compatible server on a chosen port.

- Default patient directory: example/ (resolved under --cwd, default: repo root).
- Pass specific patient JSON files with repeated --patient (relative to --patients-dir or absolute).
- Trials path: file or directory (--trials), also resolved under --cwd when relative.
- On non-zero exit, optionally re-invokes the matcher; the matcher resumes from partial JSON on disk.

Unknown CLI tokens are forwarded to match_patients_trials_qwen.py (place them after a lone '--'
if they could be mistaken for future wrapper flags).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHER = Path(__file__).resolve().parent / "match_patients_trials_qwen.py"


def _resolve_under_cwd(path: Path, cwd: Path) -> Path:
    """Resolve *path*; if relative, it is taken relative to *cwd* (not the process CWD)."""
    p = Path(path)
    return p.resolve() if p.is_absolute() else (cwd / p).resolve()


def _openai_base_url(host: str, port: int) -> str:
    """Build http(s)-style authority for OpenAI-compatible /v1 base URL (IPv6-safe)."""
    h = host.strip()
    if not h:
        raise ValueError("host must be non-empty")
    # IPv6 literals contain ':'; bracket unless already given as [addr] or [addr%zone].
    if ":" in h and not h.startswith("["):
        h = f"[{h}]"
    return f"http://{h}:{int(port)}/v1"


def _resolve_patient_files(patients_dir: Path, names: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for raw in names:
        p = Path(raw)
        if not p.is_absolute():
            p = (patients_dir / raw).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Patient JSON not found: {p}")
        if p.suffix.lower() != ".json":
            raise ValueError(f"Expected .json patient file: {p}")
        out.append(p)
    stems = [p.name for p in out]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "Duplicate patient JSON filenames (same basename) would collide in staging directory: "
            + ", ".join(sorted(stems))
        )
    return out


def _prepare_patient_staging(
    staging_root: Path, patient_paths: Sequence[Path]
) -> Path:
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / "patients_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    for src in patient_paths:
        dest = staging / src.name
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)
    return staging.resolve()


def _build_matcher_command(
    trials: Path,
    patients_path: Path,
    base_url: str,
    output_dir: Path,
    forward_args: Sequence[str],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(MATCHER),
        "--trials-dir",
        str(trials.resolve()),
        "--patients-dir",
        str(patients_path.resolve()),
        "--base-url",
        base_url.rstrip("/"),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    cmd.extend(forward_args)
    return cmd


def _forward_has_skip_failed_trials(forward: Sequence[str]) -> bool:
    i = 0
    while i < len(forward):
        tok = forward[i]
        if tok == "--skip-failed-trials":
            return True
        if tok.startswith("--skip-failed-trials="):
            return True
        i += 1
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stable batch patient–trial matching via match_patients_trials_qwen.py "
            "(resume-friendly; optional auto-restart on failure)."
        ),
        epilog=(
            "Extra flags (e.g. --model NAME, --timeout 300, --use-cot-reasoning) are passed "
            "through to match_patients_trials_qwen.py. Put them after a lone `--` if needed.\n\n"
            "Relative paths for --trials, --patients-dir, and --output-dir are resolved against "
            "--cwd (default: repository root), not the shell's current directory.\n\n"
            "Example:\n"
            "  %(prog)s --port 9220 --trials data/custom/processed_cancer_trials \\\n"
            "    --patient covid19.json --patient phenopacket.json \\\n"
            "    -- --model my-model --save-every 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="HTTP port of the OpenAI-compatible API (base URL = http://HOST:PORT/v1).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API host: hostname, IPv4, or IPv6 literal (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--patients-dir",
        type=Path,
        default=Path("example"),
        help="Directory containing patient JSON files (default: example).",
    )
    parser.add_argument(
        "--patient",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Patient JSON file: basename relative to --patients-dir, or absolute path. "
            "Repeat for multiple patients. If omitted, all *.json under --patients-dir are used."
        ),
    )
    parser.add_argument(
        "--trials",
        type=Path,
        required=True,
        help="Trial JSON file or directory of trial JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen_patient_trial_matches"),
        help="Output directory for per-patient results (default: results/qwen_patient_trial_matches).",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=REPO_ROOT,
        help=f"Anchor for relative paths and subprocess working directory (default: {REPO_ROOT}).",
    )
    parser.add_argument(
        "--auto-restart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-run the matcher after non-zero exit (default: true).",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        help=(
            "Maximum extra process launches after the first failure; 0 = unlimited. "
            "E.g. 3 allows 1 initial run plus up to 3 restarts (4 attempts total)."
        ),
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=10.0,
        help="Seconds to wait before restarting after a failed run (default: 10).",
    )
    parser.add_argument(
        "--skip-failed-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forward --skip-failed-trials to the matcher so a single bad trial "
            "does not abort the whole batch (default: true)."
        ),
    )

    args, forward = parser.parse_known_args()
    if forward and forward[0] == "--":
        forward = forward[1:]

    if not MATCHER.is_file():
        print(f"Matcher script not found: {MATCHER}", file=sys.stderr)
        return 2

    cwd = args.cwd.resolve()

    try:
        trials = _resolve_under_cwd(args.trials, cwd)
        if not trials.exists():
            raise FileNotFoundError(f"Trials path not found: {trials}")

        patients_dir = _resolve_under_cwd(args.patients_dir, cwd)
        output_dir = _resolve_under_cwd(args.output_dir, cwd)

        if args.patient:
            patient_paths = _resolve_patient_files(patients_dir, args.patient)
            patients_path = _prepare_patient_staging(output_dir, patient_paths)
        else:
            if not patients_dir.is_dir():
                raise FileNotFoundError(f"Patients directory not found: {patients_dir}")
            patients_path = patients_dir.resolve()

        base_url = _openai_base_url(args.host, args.port)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    matcher_forward: List[str] = list(forward)
    if args.skip_failed_trials and not _forward_has_skip_failed_trials(matcher_forward):
        matcher_forward.append("--skip-failed-trials")

    cmd = _build_matcher_command(
        trials=trials,
        patients_path=patients_path,
        base_url=base_url,
        output_dir=output_dir,
        forward_args=matcher_forward,
    )

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    attempt = 0
    while True:
        attempt += 1
        print(
            f"\n=== Matcher run attempt {attempt} ===\n"
            f"  base-url:   {base_url}\n"
            f"  cwd:        {cwd}\n"
            f"  trials:     {trials}\n"
            f"  patients:   {patients_path}\n"
            f"  output-dir: {output_dir}\n",
            flush=True,
        )
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except KeyboardInterrupt:
            print("\nInterrupted — exiting without further restarts.", file=sys.stderr)
            return 130

        if proc.returncode == 0:
            print("\nMatcher finished successfully.", flush=True)
            return 0

        print(
            f"\nMatcher exited with code {proc.returncode}.",
            file=sys.stderr,
            flush=True,
        )
        if not args.auto_restart:
            return proc.returncode

        if args.max_restarts > 0 and attempt > args.max_restarts:
            print(
                f"Max restarts ({args.max_restarts}) reached — giving up.",
                file=sys.stderr,
            )
            return proc.returncode

        delay = max(0.0, float(args.restart_delay))
        print(
            f"Restarting in {delay:.1f}s (attempt {attempt + 1})…",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)


if __name__ == "__main__":
    sys.exit(main())

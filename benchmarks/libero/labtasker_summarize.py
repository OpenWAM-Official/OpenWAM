#!/usr/bin/env python3
"""Summarize one LIBERO eval submission from Labtasker."""

import argparse
import sys
from pathlib import Path

import labtasker_runtime as runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queue")
    parser.add_argument("--auto-start-local-server", action="store_true")
    parser.add_argument(
        "--skip-coverage-check",
        action="store_true",
        help="Allow summary when task completeness cannot be verified (legacy or partial submissions); still validate results.",
    )
    args = parser.parse_args(argv)
    return runtime.summarize(
        args.submission_id,
        args.output_dir.expanduser().absolute(),
        queue=args.queue,
        auto_start_local_server=args.auto_start_local_server,
        skip_coverage_check=args.skip_coverage_check,
    )


if __name__ == "__main__":
    sys.exit(main())

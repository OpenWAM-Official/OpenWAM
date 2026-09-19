#!/usr/bin/env python3
"""Submit one independent LIBERO manifest or eval submission."""

import sys

import labtasker_runtime as runtime
from labtasker_runtime import build_task_inputs, submit


def main(argv=None):
    args = runtime.parse_submit_args(argv)
    submission_id = args.submission_id or runtime.new_submission_id()
    try:
        inputs = build_task_inputs(args)
    except (ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    route = args.route or f"openwam-libero-{args.operation}"
    if args.dry_run:
        print(f"[dry-run] submission={submission_id} operation={args.operation} tasks={len(inputs)}")
        return 0
    submit(
        inputs,
        submission_id=submission_id,
        route=route,
        queue=args.queue,
        max_attempts=args.max_attempts,
        priority=args.priority,
        note=args.note,
        auto_start_local_server=args.auto_start_local_server,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

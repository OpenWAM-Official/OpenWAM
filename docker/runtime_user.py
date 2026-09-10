"""Provide a writable home and NSS identity without changing the container UID."""

import os
import sys
import tempfile
from pathlib import Path


def main():
    home = Path(os.environ["HOME"])
    try:
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        # An actual write catches read-only mounts as well as UID mismatches.
        with tempfile.TemporaryFile(dir=home) as probe:
            probe.write(b"OpenWAM")
    except OSError as error:
        raise SystemExit(f"OpenWAM: home directory is not writable: {home} (check cache mount UID/GID): {error}")

    uid, gid = os.getuid(), os.getgid()
    # Retain system accounts, replacing any entry for the selected numeric ID
    # or logical runtime name. /etc/passwd and /etc/group remain untouched.
    for source, target, identity, entry in (
        ("/etc/passwd", os.environ["NSS_WRAPPER_PASSWD"], uid, f"openwam:x:{uid}:{gid}:OpenWAM:{home}:/bin/bash"),
        ("/etc/group", os.environ["NSS_WRAPPER_GROUP"], gid, f"openwam:x:{gid}:"),
    ):
        lines = [
            line
            for line in Path(source).read_text().splitlines()
            if line.split(":")[0] != "openwam" and line.split(":")[2] != str(identity)
        ]
        path = Path(target)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text("\n".join([*lines, entry]) + "\n")
        temporary.replace(path)

    # Start bash only after the account files exist: bash itself looks up the
    # current UID at startup. exec preserves signal handling and the exit code.
    os.execv("/bin/bash", ["bash", "/opt/openwam/docker/entrypoint.sh", *sys.argv[1:]])


if __name__ == "__main__":
    main()

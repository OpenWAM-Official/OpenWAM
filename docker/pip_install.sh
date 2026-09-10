#!/usr/bin/env bash
# Bootstrap cannot use uv until uv itself is installed. Honor the same trusted
# indexes here; pip searches both indexes without a primary-source preference.
set -euo pipefail
indexes=(--index-url "${PIP_INDEX_URL:-https://pypi.org/simple}")
if [[ -n "${PIP_FALLBACK_INDEX_URL:-}" ]]; then
    indexes+=(--extra-index-url "$PIP_FALLBACK_INDEX_URL")
fi
exec python -m pip install "${indexes[@]}" "$@"

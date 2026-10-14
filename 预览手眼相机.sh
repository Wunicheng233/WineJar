#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
preview_python="${WINEJAR_PYTHON:-$HOME/anaconda3/envs/winejar-xarm-sdk/bin/python}"
if [[ ! -x "$preview_python" ]]; then
    echo "Python environment not found: $preview_python; set WINEJAR_PYTHON to the SDK environment Python." >&2
    exit 1
fi
if [[ -d /usr/share/fonts/truetype/dejavu ]]; then
    export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}"
fi
exec "$preview_python" "$project_dir/xarm_sdk_control/scripts/41_preview_wrist_camera.py" "$@"

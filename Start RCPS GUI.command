#!/bin/bash
# Double-click launcher for rcps-gui (macOS; also runs on Linux via `bash`).
# Windows twin: "Start RCPS GUI.bat". Keep the two in sync.
#   1. uses ./.venv if present, otherwise the conda env named below
#   2. first run: offers to create the env and install rcps
#   3. starts the GUI with this folder as the project and opens the browser
# Close this Terminal window (or press Ctrl+C) to stop the GUI.

ENV_NAME="rcps"     # conda environment name
PORT=8421

cd "$(dirname "$0")" || exit 1
URL="http://127.0.0.1:$PORT/"

# Already running? Just reopen the browser tab.
if curl -s -o /dev/null --max-time 1 "$URL"; then
  echo "rcps-gui is already running: $URL"
  open "$URL" 2>/dev/null || xdg-open "$URL" >/dev/null 2>&1
  exit 0
fi

find_conda() {
  if [ -n "$CONDA_EXE" ] && [ -x "$CONDA_EXE" ]; then echo "$CONDA_EXE"; return; fi
  if command -v conda >/dev/null 2>&1; then command -v conda; return; fi
  for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" \
           "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" \
           /opt/homebrew/Caskroom/miniforge/base /opt/homebrew/Caskroom/miniconda/base \
           /usr/local/Caskroom/miniforge/base /usr/local/Caskroom/miniconda/base \
           /opt/miniforge3 /opt/miniconda3 /opt/anaconda3; do
    if [ -x "$d/bin/conda" ]; then echo "$d/bin/conda"; return; fi
  done
}

install_rcps() {
  echo "Installing rcps and its dependencies (a few minutes) ..."
  python -m pip install -e ".[skimage]" || { echo; echo "Install failed - see the messages above."; exit 1; }
}

ENV_DESC=""
CONDA="$(find_conda)"
if [ -x ".venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ENV_DESC=".venv in this folder"
elif [ -n "$CONDA" ]; then
  eval "$("$CONDA" shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME" && ENV_DESC="conda env '$ENV_NAME'"
  fi
fi

# First run: no environment found -> offer to create one.
if [ -z "$ENV_DESC" ]; then
  echo "No Python environment for rcps found (looked for ./.venv and conda env '$ENV_NAME')."
  read -r -p "Create one now and install rcps (needs internet)? [Y/n] " ans
  case "$ans" in [nN]*) exit 1 ;; esac
  if [ -n "$CONDA" ]; then
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11 || exit 1
    conda activate "$ENV_NAME" || exit 1
    ENV_DESC="conda env '$ENV_NAME'"
  else
    command -v python3 >/dev/null 2>&1 || { echo "Python 3 not found. Install Miniforge (https://conda-forge.org/download/) and run this again."; exit 1; }
    python3 -m venv .venv || exit 1
    # shellcheck disable=SC1091
    source .venv/bin/activate
    ENV_DESC=".venv in this folder"
  fi
  install_rcps
fi

# Environment found but packages missing -> offer to install them.
if ! python -c "import numpy, scipy, trimesh, yaml, iso2mesh, pymeshfix, rcps_gui.server" 2>/dev/null; then
  echo "Some rcps packages are missing in $ENV_DESC."
  read -r -p "Install them now (needs internet)? [Y/n] " ans
  case "$ans" in [nN]*) exit 1 ;; esac
  install_rcps
fi

echo
echo "Starting rcps-gui  [$ENV_DESC]"
echo "  project folder: $PWD"
echo "  Keep this window open while you use the GUI; close it to stop."
echo
python -m rcps_gui.server --project . --port "$PORT"

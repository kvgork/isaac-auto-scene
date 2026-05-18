#!/usr/bin/env bash
# Install Isaac Sim 5.1 and Isaac Lab v2.3.2 from NVIDIA's PyPI index.
# Run via: pixi run install-isaac  (or bash scripts/install_isaac.sh directly)
#
# Requires an active pixi environment of 'sim' or 'full'.
# Python 3.11 is required — Isaac Sim 5.1 does not support other versions.
set -euo pipefail

NVIDIA_PYPI="https://pypi.nvidia.com"
ISAACSIM_VERSION="5.1.0"
ISAACLAB_VERSION="2.3.2"

# ── environment check ──────────────────────────────────────────────────────────
ACTIVE_ENV="${PIXI_ENVIRONMENT_NAME:-}"
if [[ -z "$ACTIVE_ENV" ]]; then
    echo "WARNING: PIXI_ENVIRONMENT_NAME not set. Are you running inside a pixi shell?" >&2
fi
if [[ "$ACTIVE_ENV" != "sim" && "$ACTIVE_ENV" != "full" ]]; then
    echo "WARNING: Active pixi environment is '${ACTIVE_ENV:-<unknown>}'." >&2
    echo "         Expected 'sim' or 'full'. Isaac Sim install may land in the wrong env." >&2
    echo "         Run: pixi shell -e sim   or   pixi shell -e full" >&2
fi

# ── Isaac Sim ──────────────────────────────────────────────────────────────────
echo "Installing isaacsim==${ISAACSIM_VERSION} from ${NVIDIA_PYPI} ..."
if pip install "isaacsim==${ISAACSIM_VERSION}" --extra-index-url "${NVIDIA_PYPI}"; then
    echo "isaacsim installed successfully."
else
    echo "ERROR: pip install isaacsim failed." >&2
    echo "       NVIDIA PyPI may require an NGC account or CUDA-compatible environment." >&2
    echo "       Verify access at: ${NVIDIA_PYPI}" >&2
    exit 1
fi

# ── Isaac Lab ─────────────────────────────────────────────────────────────────
echo "Installing isaaclab==${ISAACLAB_VERSION} from ${NVIDIA_PYPI} ..."
if pip install "isaaclab==${ISAACLAB_VERSION}" --extra-index-url "${NVIDIA_PYPI}"; then
    echo "isaaclab installed successfully."
else
    echo "pip install isaaclab failed — falling back to source install." >&2
    echo "" >&2
    echo "Run the following commands to install Isaac Lab from source:" >&2
    echo "  git clone https://github.com/isaac-sim/IsaacLab.git" >&2
    echo "  cd IsaacLab" >&2
    echo "  git checkout v${ISAACLAB_VERSION}" >&2
    echo "  ./isaaclab.sh -i" >&2
    echo "" >&2
    echo "See also: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/" >&2
    exit 1
fi

echo ""
echo "Isaac Sim ${ISAACSIM_VERSION} + Isaac Lab ${ISAACLAB_VERSION} installed."
echo "Verify with: python -c \"import isaacsim; print(isaacsim.__version__)\""

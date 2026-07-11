#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.wsl_env"

if [[ "$ROOT_DIR" == *" "* ]]; then
  cat >&2 <<EOF
[error] This project path contains spaces:
  $ROOT_DIR

Vortex Makefiles split paths containing spaces. Copy the project into a
WSL-native path without spaces, then rerun this script:

  mkdir -p ~/hackathon-project
  rsync -a --delete --exclude '.git/' --exclude 'vendor/vortex/.git/' "$ROOT_DIR/" ~/hackathon-project/
  cd ~/hackathon-project
  bash scripts/wsl_run_agent_artifacts.sh
EOF
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

cd "$ROOT_DIR"

# Keep copied worktrees relocatable. A .wsl_env generated under /mnt/c may have
# stale VORTEX_* paths after rsync into ~/hackathon-project.
export VORTEX_HOME="$ROOT_DIR/vendor/vortex"
export VORTEX_BUILD_DIR="$VORTEX_HOME/build"
export VORTEX_PATH="$VORTEX_BUILD_DIR/install"
export PKG_CONFIG_PATH="$VORTEX_PATH/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

missing_prereqs=()
[[ -f "$VORTEX_HOME/third_party/ramulator/src/base/base.h" ]] || missing_prereqs+=("ramulator header: vendor/vortex/third_party/ramulator/src/base/base.h")
[[ -f "$VORTEX_HOME/third_party/softfloat/source/include/softfloat_types.h" ]] || missing_prereqs+=("softfloat header: vendor/vortex/third_party/softfloat/source/include/softfloat_types.h")
[[ -f "$VORTEX_BUILD_DIR/config.mk" ]] || missing_prereqs+=("Vortex build config: vendor/vortex/build/config.mk")

if [[ "${#missing_prereqs[@]}" -ne 0 ]]; then
  echo "[not-ready] Vortex sim prerequisites are missing:" >&2
  printf '  - %s\n' "${missing_prereqs[@]}" >&2
  echo >&2
  echo "Run this from $ROOT_DIR:" >&2
  echo "  bash scripts/wsl_setup_vortex.sh" >&2
  echo "  source .wsl_env" >&2
  echo "  bash scripts/wsl_run_agent_artifacts.sh" >&2
  exit 1
fi

python3 grow_compiler.py \
  --provider mock \
  --program examples/demo_program_full.ir \
  --rules data/rules.local.json \
  --emit-vortex-tests artifacts/vortex_tests

for dir in artifacts/vortex_tests/agent_*; do
  [[ -d "$dir" ]] || continue
  echo "[simx] $dir"
  make -C "$dir" run-simx
done

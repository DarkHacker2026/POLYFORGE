#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VORTEX_HOME="$ROOT_DIR/vendor/vortex"
TOOLDIR="${TOOLDIR:-$HOME/tools}"
XLEN="${XLEN:-32}"
VORTEX_OSVERSION="${VORTEX_OSVERSION:-ubuntu/focal}"

echo "[root] $ROOT_DIR"
echo "[vortex] $VORTEX_HOME"
echo "[tooldir] $TOOLDIR"
echo "[osversion] $VORTEX_OSVERSION"

if [[ "$ROOT_DIR" == *" "* ]]; then
  cat >&2 <<EOF
[error] This project path contains spaces:
  $ROOT_DIR

Vortex Makefiles do not reliably quote internal paths. Move/copy the project to
a WSL-native path without spaces, for example:

  mkdir -p ~/hackathon-project
  rsync -a --delete --exclude '.git/' --exclude 'vendor/vortex/.git/' "$ROOT_DIR/" ~/hackathon-project/
  cd ~/hackathon-project
  bash scripts/wsl_setup_vortex.sh
EOF
  exit 1
fi

if [[ ! -d "$VORTEX_HOME" ]]; then
  echo "[error] Vortex repo not found at $VORTEX_HOME" >&2
  exit 1
fi

write_env_file() {
  echo "[env] writing $ROOT_DIR/.wsl_env"
  cat > "$ROOT_DIR/.wsl_env" <<EOF
export TOOLDIR="$TOOLDIR"
export VORTEX_HOME="$VORTEX_HOME"
export VORTEX_BUILD_DIR="$VORTEX_HOME/build"
export VORTEX_PATH="$VORTEX_HOME/build/install"
export PKG_CONFIG_PATH="\$VORTEX_PATH/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
export PATH="$TOOLDIR/llvm-vortex/bin:$TOOLDIR/riscv32-gnu-toolchain/bin:$TOOLDIR/verilator/bin:\$PATH"
EOF
}

write_env_file

echo "[normalize] fixing Windows CRLF line endings in Vortex scripts"
find "$VORTEX_HOME" \
  -type d \( -name ".git" -o -name "third_party" \) -prune -o \
  \( -name configure -o -name VERSION -o -name "*.sh" -o -name "*.sh.in" -o -name "*.py" \) \
  -type f -print0 | xargs -0 sed -i 's/\r$//'

ensure_git_checkout() {
  local path="$1"
  local url="$2"
  local marker="$3"
  local ref="${4:-}"
  if [[ -e "$path/$marker" ]]; then
    return
  fi
  if [[ -d "$path/.git" ]]; then
    echo "[third_party] updating $path"
    git -C "$path" remote set-url origin "$url" || true
    if [[ -n "$ref" ]]; then
      git -C "$path" fetch --depth 1 origin "$ref" || git -C "$path" fetch origin "$ref" || true
      git -C "$path" checkout -f "$ref" || true
    else
      git -C "$path" fetch --depth 1 origin || true
      git -C "$path" checkout -f FETCH_HEAD || true
    fi
    git -C "$path" submodule update --init --recursive || true
    if [[ -e "$path/$marker" ]]; then
      return
    fi
  fi
  if [[ -d "$path" ]]; then
    local backup="$path.incomplete.$(date +%s)"
    echo "[third_party] $path missing $marker; moving to $backup"
    mv "$path" "$backup"
  fi
  echo "[third_party] cloning $url -> $path"
  rm -rf "$path"
  git clone --recursive "$url" "$path"
  if [[ -n "$ref" ]]; then
    git -C "$path" fetch origin "$ref" || true
    git -C "$path" checkout -f "$ref"
    git -C "$path" submodule update --init --recursive
  fi
  if [[ ! -e "$path/$marker" ]]; then
    echo "[error] cloned $url but still missing $marker" >&2
    echo "[debug] candidate headers:" >&2
    find "$path" -name base.h -o -name softfloat_types.h >&2 || true
    exit 1
  fi
}

echo "[third_party] ensuring source dependencies"
ensure_git_checkout \
  "$VORTEX_HOME/third_party/softfloat" \
  "https://github.com/ucb-bar/berkeley-softfloat-3.git" \
  "source/include/softfloat_types.h" \
  "b51ef8f3201669b2288104c28546fc72532a1ea4"
ensure_git_checkout \
  "$VORTEX_HOME/third_party/ramulator" \
  "https://github.com/CMU-SAFARI/ramulator2.git" \
  "src/base/base.h" \
  "e62c84a6f0e06566ba6e182d308434b4532068a5"

echo "[apt] installing host packages"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ccache \
  git \
  wget \
  tar \
  bzip2 \
  python3 \
  python3-venv \
  python3-pip \
  pkg-config \
  zlib1g-dev \
  libtinfo-dev \
  libncurses-dev \
  uuid-dev \
  libboost-serialization-dev \
  libpng-dev \
  libhwloc-dev \
  libffi-dev \
  opencl-headers \
  ocl-icd-opencl-dev

echo "[configure] Vortex build"
cd "$VORTEX_HOME"
mkdir -p build
cd build
../configure --xlen="$XLEN" --tooldir="$TOOLDIR" --osversion="$VORTEX_OSVERSION"

echo "[configure] generating ci/toolchain_install.sh from template"
# shellcheck disable=SC1091
source "$VORTEX_HOME/VERSION"
TOOLCHAIN_REV="${TOOLCHAIN_REV//$'\r'/}"
VORTEX_VERSION="${VORTEX_VERSION//$'\r'/}"
GEM5_REV="${GEM5_REV//$'\r'/}"
mkdir -p ./ci
sed "s|@VORTEX_HOME@|$VORTEX_HOME|g; s|@XLEN@|$XLEN|g; s|@TOOLDIR@|$TOOLDIR|g; s|@OSVERSION@|$VORTEX_OSVERSION|g; s|@INSTALLDIR@|$VORTEX_HOME/build/install|g; s|@BUILDDIR@|$VORTEX_HOME/build|g; s|@TOOLCHAIN_REV@|$TOOLCHAIN_REV|g; s|@VORTEX_VERSION@|$VORTEX_VERSION|g; s|@GEM5_REV@|$GEM5_REV|g" \
  "$VORTEX_HOME/ci/toolchain_install.sh.in" > ./ci/toolchain_install.sh
sed -i 's/\r$//' ./ci/toolchain_install.sh
chmod +x ./ci/toolchain_install.sh

echo "[toolchain] installing minimal Vortex toolchain for kernel tests"
missing_tools=()
[[ -x "$TOOLDIR/riscv32-gnu-toolchain/bin/riscv32-unknown-elf-gcc" ]] || missing_tools+=(--riscv32)
[[ -x "$TOOLDIR/llvm-vortex/bin/clang" ]] || missing_tools+=(--llvm)
[[ -f "$TOOLDIR/libcrt32/lib/baremetal/libclang_rt.builtins-riscv32.a" ]] || missing_tools+=(--libcrt32)
[[ -d "$TOOLDIR/libc32" ]] || missing_tools+=(--libc32)
[[ -x "$TOOLDIR/verilator/bin/verilator" ]] || missing_tools+=(--verilator)

if [[ "${#missing_tools[@]}" -eq 0 ]]; then
  echo "[toolchain] required tools already present; skipping downloads"
else
  ./ci/toolchain_install.sh "${missing_tools[@]}"
fi

echo "[build] Vortex kernel/runtime/simx pieces"
if [[ ! -f Makefile ]]; then
  echo "[warn] Vortex build Makefile missing after configure; generated tests may still build simulator targets on demand."
else
  make -C "$VORTEX_HOME/third_party" softfloat ramulator
  make -C sim/simx
  make -C sw/runtime/simx
fi

write_env_file

echo "[done] WSL Vortex setup complete"
echo "Next:"
echo "  source \"$ROOT_DIR/.wsl_env\""
echo "  python3 tools/check_env.py"
echo "  make -C artifacts/vortex_tests/agent_add_add_int_v1_instantiated run-simx"

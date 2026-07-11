#!/usr/bin/env bash
set -e
STAGED=~/hackathon-project/vendor/vortex/build/tests/kernel/simt_probe_lmem
mkdir -p "$STAGED"
cp ~/hackathon-project/artifacts/lmem_probe/simt_probe_lmem/main.cpp "$STAGED/main.cpp"

cat > "$STAGED/Makefile" << 'MAKEOF'
ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := simt_probe_lmem
SRC_DIR := $(VORTEX_BUILD_DIR)/tests/kernel/$(PROJECT)
SRCS    := $(SRC_DIR)/main.cpp

include $(VORTEX_HOME)/tests/kernel/common.mk
MAKEOF

echo "staged ok"
ls "$STAGED/"

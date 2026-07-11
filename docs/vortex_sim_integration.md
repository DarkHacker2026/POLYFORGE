# Vortex Simulator Integration Plan

The first scaffold uses `LocalSemanticSimulator` so the agent loop is runnable immediately. The next verification backend should use Vortex's existing low-level binary flow, not OpenCL/HIP/Vulkan.

## What To Reuse

Vortex kernel tests use:

```text
vendor/vortex/tests/kernel/common.mk
vendor/vortex/sw/kernel/scripts/link32.ld
vendor/vortex/sw/kernel/scripts/vxbin.py
vendor/vortex/sim/simx/simx
vendor/vortex/sim/rtlsim/rtlsim
```

The important flow from `tests/kernel/common.mk` is:

```text
kernel source/assembly
  -> clang --target=riscv32-unknown-elf -march=rv32imaf -mabi=ilp32f +xvortex
  -> ELF linked at STARTUP_ADDR=0x80000000
  -> vxbin.py converts ELF to .vxbin
  -> simx or rtlsim executes .vxbin
```

## What To Avoid

Do not route generated operations through:

```text
OpenCL
HIP
Vulkan
POCL source lowering
```

Using an assembler/linker as a mechanical encoder is fine. Using an existing high-level compiler to lower the operation is not.

## Next Implementation Step

Add an emitter that turns a verified candidate like:

```json
[
  {"op": "ADD", "dst": "r3", "src1": "r1", "src2": "r2"}
]
```

into a minimal Vortex kernel test directory:

```text
artifacts/vortex_tests/add_direct_v1/
  Makefile
  main.cpp
```

Then run:

```bash
make -C artifacts/vortex_tests/agent_add_add_direct_v1 run-simx
```

The generated Makefile stages itself into
`vendor/vortex/build/tests/kernel/agent_*` because Vortex's
`tests/kernel/common.mk` assumes `../../..` points at the configured build tree
containing `config.mk`.

The simulator result should be parsed into the same proof shape already used by the local simulator:

```json
{
  "ok": true,
  "cycles": 123,
  "simulator": "vortex_simx"
}
```

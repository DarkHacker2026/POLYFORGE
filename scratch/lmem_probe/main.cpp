#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

// Probe: read VX_CSR_LOCAL_MEM_BASE to confirm the scratchpad base address,
// then write/readback a sentinel value to prove it is physically accessible.
//
// Sources:
//   VX_CSR_LOCAL_MEM_BASE  = 0xFC3  (vendor/vortex/build/sw/VX_types.h:324)
//   VX_MEM_LMEM_BASE_ADDR  = 4294901760 = 0xFFFF0000 (VX_types.h:25)
//   VX_CFG_LMEM_LOG_SIZE   = 14   (vendor/vortex/build/sw/VX_config.h:946)
//   VX_CFG_LMEM_ENABLED    = 1    (VX_config.h:97)

int main() {
  // Step 1: read base address from CSR (the real hardware-reported value)
  uint32_t lmem_base = (uint32_t)csr_read(VX_CSR_LOCAL_MEM_BASE);

  // Step 2: compute scratchpad size from compile-time macro (not runtime)
  uint32_t lmem_size = (1u << VX_CFG_LMEM_LOG_SIZE);

  // Step 3: write sentinel, read back, confirm round-trip
  volatile int32_t* lmem_ptr = (volatile int32_t*)lmem_base;
  lmem_ptr[0] = 57005;
  int32_t readback = lmem_ptr[0];

  // SIMX_RESULT = upper 16 bits of base addr (expected 65535 = 0xFFFF)
  int result   = (int)(lmem_base >> 16);
  int expected = 65535;

  vx_printf("LMEM_BASE_HI=%d\n", (int)(lmem_base >> 16));
  vx_printf("LMEM_BASE_LO=%d\n", (int)(lmem_base & 0xFFFF));
  vx_printf("LMEM_SIZE=%d\n",    (int)lmem_size);
  vx_printf("LMEM_READBACK=%d\n",(int)readback);
  vx_printf("SIMX_RESULT=%d\n",  result);
  vx_printf("SIMX_EXPECTED=%d\n",expected);
  vx_printf("SIMX_CYCLES=1\n");
  if (result == expected && readback == 57005) {
    vx_printf("Passed! result matched expected\n");
  } else {
    vx_printf("FAILED\n");
  }
  return 0;
}

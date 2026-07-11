#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

volatile int out[16] = {0};
uint32_t N = 8;

static void kernel(void *__args) {
    int i = blockIdx.x;
    int j = vx_thread_id();
    int w = vx_warp_id();
    vx_printf("Warp %d, Thread %d, BlockIdx %d\n", w, j, i);
    out[i] = i + 1;
}

int main() {
    vx_spawn_threads(1, &N, nullptr, kernel, nullptr);
    for (int i=0; i<N; i++) {
        vx_printf("out[%d] = %d\n", i, out[i]);
    }
    return 0;
}

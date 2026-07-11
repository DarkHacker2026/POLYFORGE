// doom_raycaster.cu — Micro-DOOM Raycaster for POLYFORGE
// A simple CUDA raycaster that renders a 2D map to ASCII art.
// Each thread represents one vertical column of the screen.

#define SCREEN_WIDTH 64
#define MAP_WIDTH 8
#define MAP_HEIGHT 8
#define MAX_DEPTH 16.0f
#define FOV 3.14159f / 3.0f

// 8x8 map: 1 = wall, 0 = empty
__device__ const int d_map[64] = {
    1,1,1,1,1,1,1,1,
    1,0,0,0,0,0,0,1,
    1,0,1,1,0,1,0,1,
    1,0,1,0,0,1,0,1,
    1,0,1,0,0,0,0,1,
    1,0,0,0,1,1,0,1,
    1,0,0,0,0,0,0,1,
    1,1,1,1,1,1,1,1
};

__global__ void render_doom(float player_x, float player_y, float player_a, float fov, int screen_width, float max_depth, int* output_buffer) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < screen_width) {
        float ray_angle = (player_a - fov / 2.0f) + ((float)x / (float)screen_width) * fov;
        float ray_dx = cosf(ray_angle);
        float ray_dy = sinf(ray_angle);
        float distance = 0.0f;
        float step_size = 0.1f;
        int hit = 0;
        while (distance < max_depth && hit == 0) {
            float test_x = player_x + ray_dx * distance;
            float test_y = player_y + ray_dy * distance;
            int map_x = (int)test_x;
            int map_y = (int)test_y;
            if (map_x < 0 || map_x >= MAP_WIDTH || map_y < 0 || map_y >= MAP_HEIGHT) {
                hit = 1;
            } else {
                int cell = d_map[map_y * MAP_WIDTH + map_x];
                if (cell == 1) {
                    hit = 1;
                } else {
                    distance += step_size;
                }
            }
        }
        int intensity = 0;
        if (hit == 1) {
            float normalized = 1.0f - (distance / max_depth);
            if (normalized < 0.0f) normalized = 0.0f;
            if (normalized > 1.0f) normalized = 1.0f;
            intensity = (int)(normalized * 100.0f);
        }
        output_buffer[x] = intensity;
    }
}

int main() {
    float player_x = 4.0f;
    float player_y = 4.0f;
    float player_a = 0.0f;
    float fov = FOV;
    int screen_width = SCREEN_WIDTH;
    float max_depth = MAX_DEPTH;

    int output_buffer[SCREEN_WIDTH];
    for (int i = 0; i < SCREEN_WIDTH; i++) {
        output_buffer[i] = 0;
    }

    render_doom<<<1, SCREEN_WIDTH>>>(player_x, player_y, player_a, fov, screen_width, max_depth, output_buffer);
    cudaDeviceSynchronize();

    char line[SCREEN_WIDTH + 1];
    for (int i = 0; i < SCREEN_WIDTH; i++) {
        int val = output_buffer[i];
        if (val > 75) line[i] = '#';
        else if (val > 50) line[i] = 'x';
        else if (val > 25) line[i] = '.';
        else line[i] = ' ';
    }
    line[SCREEN_WIDTH] = '\0';

    printf("=== MICRO-DOOM RAYCASTER ===\n");
    printf("Player: (%.1f, %.1f) Angle: %.1f rad\n", player_x, player_y, player_a);
    printf("FOV: %.1f rad  Depth: %.1f  Width: %d\n", fov, max_depth, screen_width);
    printf("\n%s\n\n", line);
    printf("============================\n");

    return 0;
}
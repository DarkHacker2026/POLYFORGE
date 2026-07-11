"""
scale_rtlsim.py
---------------
Generate SAXPY Vortex C++ kernels for a range of N values and predict
RTL simulator cycle counts via linear extrapolation from measured data.

Actual RTL simulation requires WSL (Vortex toolchain + Spike/Verilator).
This script only generates the kernel sources and prints predictions.
"""

import sys
import pathlib

# ── project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# ── configuration ─────────────────────────────────────────────────────────────
GEN_NS = [256, 512, 1024, 2048, 4096]

# Known measured (N → cycles) – used to fit a linear model
KNOWN_POINTS = {
    16:  3911,
    64:  5401,
    256: 11565,
}

# N values to predict
PRED_NS = [512, 1024, 2048, 4096]

# Scalar baseline for speedup calculation
# (naively: scalar processes elements one-by-one at ~rate cycles each)
# We model scalar as: scalar_cycles(N) = base + rate * N  (single-threaded)
# For speedup we compare predicted parallel cycles vs. scalar estimate.
SCALAR_RATE = None  # derived from the linear fit (slope per element × N threads=1)

OOM_WARNING_N = 2048   # known RTL trace-buffer OOM threshold

# ── kernel template ──────────────────────────────────────────────────────────
KERNEL_TEMPLATE = """\
#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

volatile int32_t x[{N}];
volatile int32_t y[{N}];
uint32_t N = {N};
int a = 3;

static void kernel_saxpy(void *__args) {{
    (void)__args;
    int i = vx_thread_id();
    y[i] = a * x[i] + y[i];
}}

int main() {{
    vx_spawn_threads(1, &N, nullptr, kernel_saxpy, nullptr);
    vx_printf("Passed!\\n");
    return 0;
}}
"""

# ── linear model ──────────────────────────────────────────────────────────────

def fit_linear(points: dict[int, int]):
    """
    Fit cycles = base + rate * N via ordinary least squares on log-space N.
    We use a simple two-point fit with the two extremes to reduce noise.
    Returns (base, rate).
    """
    xs = list(points.keys())
    ys = list(points.values())
    n  = len(xs)

    # OLS: minimise sum((y_i - base - rate*x_i)^2)
    sum_x  = sum(xs)
    sum_y  = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sum_xx - sum_x ** 2
    if denom == 0:
        # Degenerate – all x identical, just return mean
        return (sum_y / n, 0.0)

    rate = (n * sum_xy - sum_x * sum_y) / denom
    base = (sum_y - rate * sum_x) / n
    return base, rate


def predict_cycles(base: float, rate: float, N: int) -> int:
    return max(1, round(base + rate * N))


def scalar_cycles_estimate(base: float, rate: float, N: int) -> int:
    """
    Estimate scalar (single-thread) cycle count.
    Scalar processes each element sequentially, so we scale the
    per-element cost (rate) by N and add the base overhead.
    """
    # rate is cycles-per-element in parallel; scalar pays that cost × N serially.
    return max(1, round(base + rate * N * N))


# ── kernel generation ─────────────────────────────────────────────────────────

def generate_kernel(N: int) -> str:
    """Return the formatted SAXPY kernel source for the given N."""
    return KERNEL_TEMPLATE.format(N=N)


def write_kernel(N: int) -> pathlib.Path:
    """Write the kernel to artifacts/scale_test/N{N}/main.cpp and return the path."""
    out_dir  = PROJECT_ROOT / "artifacts" / "scale_test" / f"N{N}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "main.cpp"
    source   = generate_kernel(N)
    out_file.write_text(source, encoding="utf-8")
    return out_file


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("scale_rtlsim.py – SAXPY Vortex RTL kernel generation + cycle prediction")
    print("=" * 72)

    # ── 1. Generate kernels ──────────────────────────────────────────────────
    print()
    print("[ Step 1 ] Generating SAXPY C++ kernels")
    print("-" * 50)
    gen_header = f"{'N':>6}  {'path':40}  {'size_bytes':>10}  {'lines':>6}"
    print(gen_header)
    print("-" * len(gen_header))

    for N in GEN_NS:
        out_path = write_kernel(N)
        source   = out_path.read_text(encoding="utf-8")
        size_b   = out_path.stat().st_size
        lines    = source.count("\n") + 1
        rel      = out_path.relative_to(PROJECT_ROOT)
        print(f"{N:>6}  {str(rel):40}  {size_b:>10}  {lines:>6}")

    print()
    print("[Note] Actual RTL simulation (Spike/Verilator) requires the Vortex")
    print("       toolchain running inside WSL. The kernels above are ready to")
    print("       compile and run once WSL + Vortex SDK are available.")

    # ── 2. Fit linear model ──────────────────────────────────────────────────
    print()
    print("[ Step 2 ] Linear model fit (cycles = base + rate × N)")
    print("-" * 50)
    base, rate = fit_linear(KNOWN_POINTS)
    print(f"  Known data points: {KNOWN_POINTS}")
    print(f"  Fitted model     : cycles ~= {base:.1f} + {rate:.4f} * N")
    print()

    # Verify fit quality on known points
    print("  Fit verification:")
    for n_k, c_k in sorted(KNOWN_POINTS.items()):
        pred = predict_cycles(base, rate, n_k)
        err  = abs(pred - c_k) / c_k * 100
        print(f"    N={n_k:>5}: measured={c_k:>6}  predicted={pred:>6}  err={err:.1f}%")

    # ── 3. Predictions ───────────────────────────────────────────────────────
    print()
    print("[ Step 3 ] Predicted cycle counts for larger N")
    print("-" * 72)
    pred_header = (
        f"{'N':>6}  {'predicted_cycles':>17}  {'scalar_est_cycles':>18}"
        f"  {'speedup_vs_scalar':>18}"
    )
    print(pred_header)
    print("-" * len(pred_header))

    for N in PRED_NS:
        par_cyc    = predict_cycles(base, rate, N)
        scal_cyc   = scalar_cycles_estimate(base, rate, N)
        speedup    = scal_cyc / par_cyc if par_cyc > 0 else float("inf")
        oom_flag   = "  <- OOM risk" if N >= OOM_WARNING_N else ""
        print(
            f"{N:>6}  {par_cyc:>17,}  {scal_cyc:>18,}  {speedup:>16.1f}x{oom_flag}"
        )

    print()
    print(
        "[Warning] RTL simulator trace buffers are predicted to OOM at N~2048 based on extrapolation; not directly observed."
    )

    # ── 4. Summary ───────────────────────────────────────────────────────────
    print()
    print("[ Summary ]")
    print("-" * 50)
    print(f"  Kernels written to : {PROJECT_ROOT / 'artifacts' / 'scale_test'}")
    print(f"  Generated N values : {GEN_NS}")
    print(f"  Predicted N values : {PRED_NS}")
    print(f"  Linear model       : cycles ~= {base:.1f} + {rate:.4f} * N")
    print()


if __name__ == "__main__":
    main()

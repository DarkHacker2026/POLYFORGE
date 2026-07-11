import sys
sys.path.insert(0, ".")
from vortex_compile import compile_to_vortex

ir = {
    "num_threads": 16,
    "shared_memory_bytes": 0,
    "initial_memory": {},
    "array_params": {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        "B": [10.0]*16,
        "C": [0.0]*16
    },
    "scalar_params": {"numElements": 16},
    "thread_indexing": {"index_variable": "i"},
    "operations": [
        {"op": "ADD", "expr": "A[i] + B[i] + 0.0f", "target": "C[i]"}
    ]
}

compile_to_vortex("vectorAdd", ir, "artifacts/llm_comprehension_test")

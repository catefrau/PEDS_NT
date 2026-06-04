import time
from contextlib import contextmanager
from collections import defaultdict


# ─────────────────────────────────────────────
# SECTION 2: Utils: timings, geometry updates, sanity checks
# ─────────────────────────────────────────────

# Global registry — stores all measured times
_TIMINGS = defaultdict(list)

@contextmanager
def timer(label: str, verbose: bool = True):
    """
    Context manager that measures wall time and stores it.
    Usage:  with timer("forward pass"):
                result = model(x)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _TIMINGS[label].append(elapsed)
    if verbose:
        print(f"  ⏱  {label:<35s} {elapsed*1000:.1f} ms")

def print_timing_report():
    """Print a summary of all measured times after training."""
    print("\n" + "="*60)
    print(f"{'TIMING REPORT':^60}")
    print("="*60)
    print(f"{'Step':<35s} {'calls':>6s} {'total(s)':>10s} {'mean(ms)':>10s} {'min(ms)':>10s}")
    print("-"*60)
    for label, times in sorted(_TIMINGS.items()):
        total   = sum(times)
        mean_ms = (total / len(times)) * 1000
        min_ms  = min(times) * 1000
        print(f"{label:<35s} {len(times):>6d} {total:>10.2f} {mean_ms:>10.1f} {min_ms:>10.1f}")
    print("="*60)

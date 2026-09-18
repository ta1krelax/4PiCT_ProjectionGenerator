"""Adaptive search for the projection count N that hits a target SSIM.

Pure algorithm, no GPU/Qt dependency, so it can be unit-tested against a synthetic
SSIM(N) curve before wiring it to the (expensive) real generate->reconstruct->SSIM
pipeline. Strategy: start at n_init, double/halve to bracket the target (SSIM(N) is
assumed roughly monotonic increasing in N), then bisect the integer bracket. Always
returns the best point actually evaluated, not just wherever bisection lands --
sampling-scheme noise can make SSIM(N) non-monotonic near the target.
"""

from __future__ import annotations


def _clip(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def search_n_for_target_ssim(
    evaluate,
    n_init: int,
    target_ssim: float,
    n_min: int = 10,
    n_max: int = 5000,
    ssim_tol: float = 0.01,
    n_tol: int = 5,
    max_evals: int = 20,
    should_stop=None,
    on_eval=None,
) -> dict:
    """evaluate(n:int) -> ssim:float. Returns
    {"best_n", "best_ssim", "history": [{"n","ssim"}...], "stopped_early": bool}.
    """
    history: list[dict] = []
    stopped_early = False

    def _eval(n: int) -> float:
        n = _clip(int(n), n_min, n_max)
        for h in history:
            if h["n"] == n:
                return h["ssim"]
        s = evaluate(n)
        history.append({"n": n, "ssim": s})
        if on_eval is not None:
            on_eval(n, s)
        return s

    def _cancelled() -> bool:
        return bool(should_stop and should_stop())

    n0 = _clip(n_init, n_min, n_max)
    s0 = _eval(n0)

    if abs(s0 - target_ssim) <= ssim_tol:
        return _finish(history, target_ssim, stopped_early)

    lo_n = hi_n = None
    if s0 < target_ssim:
        lo_n, lo_s = n0, s0
        n = n0
        while len(history) < max_evals and not _cancelled():
            n_next = n * 2
            if n_next >= n_max:
                s = _eval(n_max)
                if s < target_ssim:
                    lo_n, lo_s = n_max, s
                else:
                    hi_n, hi_s = n_max, s
                break
            n = n_next
            s = _eval(n)
            if abs(s - target_ssim) <= ssim_tol:
                return _finish(history, target_ssim, stopped_early)
            if s >= target_ssim:
                hi_n, hi_s = n, s
                break
            lo_n, lo_s = n, s
        else:
            if _cancelled():
                stopped_early = True
    else:
        hi_n, hi_s = n0, s0
        n = n0
        while len(history) < max_evals and not _cancelled():
            n_next = n // 2
            if n_next <= n_min:
                s = _eval(n_min)
                if s > target_ssim:
                    hi_n, hi_s = n_min, s
                else:
                    lo_n, lo_s = n_min, s
                break
            n = n_next
            s = _eval(n)
            if abs(s - target_ssim) <= ssim_tol:
                return _finish(history, target_ssim, stopped_early)
            if s <= target_ssim:
                lo_n, lo_s = n, s
                break
            hi_n, hi_s = n, s
        else:
            if _cancelled():
                stopped_early = True

    if lo_n is None or hi_n is None:
        # bounds exhausted without bracketing the target (target unreachable within
        # [n_min, n_max]) -- report the best point actually evaluated.
        return _finish(history, target_ssim, stopped_early)

    while len(history) < max_evals and (hi_n - lo_n) > n_tol and not _cancelled():
        mid = (lo_n + hi_n) // 2
        if mid == lo_n or mid == hi_n:
            break
        s = _eval(mid)
        if abs(s - target_ssim) <= ssim_tol:
            break
        if s < target_ssim:
            lo_n = mid
        else:
            hi_n = mid
    if _cancelled():
        stopped_early = True

    return _finish(history, target_ssim, stopped_early)


def _finish(history: list[dict], target_ssim: float, stopped_early: bool) -> dict:
    best = min(history, key=lambda h: abs(h["ssim"] - target_ssim))
    return {
        "best_n": best["n"],
        "best_ssim": best["ssim"],
        "history": history,
        "stopped_early": stopped_early,
    }

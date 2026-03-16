import numpy as np


def compute_rtg(stepwise_returns: np.ndarray, done_idxs: np.ndarray, gamma: float) -> np.ndarray:
    rtg = np.zeros_like(stepwise_returns, dtype=np.float32)
    start = 0
    for end in done_idxs:
        r = stepwise_returns[start:end]
        if np.isclose(gamma, 1.0, rtol=1e-09, atol=1e-09):
            rtg[start:end] = np.cumsum(r[::-1], axis=0)[::-1]
        else:
            out = np.empty_like(r, dtype=np.float32)
            running = 0.0
            for i in range(len(r) - 1, -1, -1):
                running = r[i] + gamma * running
                out[i] = running
            rtg[start:end] = out
        start = end

    return rtg


def ob_to_float(observations: np.ndarray) -> np.ndarray:
    if observations.dtype == np.uint8:
        return observations.astype(np.float32) / 255.0
    return observations.astype(np.float32)
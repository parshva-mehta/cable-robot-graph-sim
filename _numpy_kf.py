"""Pure NumPy drop-in replacement for gtsam.KalmanFilter and gtsam.noiseModel.

Provides the same API surface used by ekf.py and ekf_alt.py so that GTSAM
(which segfaults on this system) is not required.

Public API (mirrors gtsam usage in ekf.py / ekf_alt.py):
    KalmanFilter(n)
    noiseModel.Diagonal.Sigmas(sigmas)
    GaussianDensity  (returned by KalmanFilter methods, not constructed directly)
"""

import numpy as np


class _DiagNoise:
    def __init__(self, sigmas: np.ndarray):
        self.sigmas = np.asarray(sigmas, dtype=np.float64).reshape(-1)


class _DiagNoiseFactory:
    @staticmethod
    def Sigmas(sigmas) -> _DiagNoise:
        return _DiagNoise(sigmas)


class _NoiseModelNamespace:
    Diagonal = _DiagNoiseFactory


noiseModel = _NoiseModelNamespace


class GaussianDensity:
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self._mean = np.asarray(mean, dtype=np.float64).reshape(-1, 1)
        self._cov  = np.asarray(cov,  dtype=np.float64)

    def mean(self) -> np.ndarray:
        return self._mean

    def covariance(self) -> np.ndarray:
        return self._cov


class KalmanFilter:
    def __init__(self, n: int):
        self.n = n

    def init(self, x0: np.ndarray, P0: np.ndarray) -> GaussianDensity:
        return GaussianDensity(x0, P0)

    def predict(self, state: GaussianDensity,
                F: np.ndarray, B: np.ndarray, u: np.ndarray,
                model_q: _DiagNoise) -> GaussianDensity:
        x = state._mean
        P = state._cov
        Q = np.diag(model_q.sigmas ** 2)
        x_pred = F @ x + B @ np.asarray(u, dtype=np.float64).reshape(-1, 1)
        P_pred = F @ P @ F.T + Q
        return GaussianDensity(x_pred, P_pred)

    def update(self, state: GaussianDensity,
               H: np.ndarray, z: np.ndarray,
               model_r: _DiagNoise) -> GaussianDensity:
        x = state._mean
        P = state._cov
        R = np.diag(model_r.sigmas ** 2)
        z_col = np.asarray(z, dtype=np.float64).reshape(-1, 1)

        S = H @ P @ H.T + R
        # Solve S @ K.T = H @ P  (more stable than K = P@H.T @ inv(S))
        try:
            K = np.linalg.solve(S, H @ P).T
        except np.linalg.LinAlgError:
            K = P @ H.T @ np.linalg.pinv(S)

        innovation = z_col - H @ x
        x_post = x + K @ innovation

        # Joseph form: numerically stable even when K is approximate
        IKH = np.eye(self.n) - K @ H
        P_post = IKH @ P @ IKH.T + K @ R @ K.T

        return GaussianDensity(x_post, P_post)

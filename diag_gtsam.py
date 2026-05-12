import gtsam
import numpy as np

print("GTSAM version:", getattr(gtsam, "__version__", "unknown"))

# Test 1: basic 36-dim init with scaled identity
kf = gtsam.KalmanFilter(36)
mean = np.zeros((36, 1))
P = np.eye(36) * 1e-4
s = kf.init(mean, P)
print("Test 1 (I*1e-4) ok, mean norm:", np.linalg.norm(s.mean()))

# Test 2: large-ish P
P2 = np.eye(36) * 1e6
s2 = kf.init(mean, P2)
print("Test 2 (I*1e6) ok")

# Test 3: P at max bound
P3 = np.eye(36) * 1e8
s3 = kf.init(mean, P3)
print("Test 3 (I*1e8) ok")

# Test 4: simulate what happens after predict + reinit_jitter
F = np.eye(36)
B = np.eye(36)
u = np.zeros((36, 1))
Q = gtsam.noiseModel.Diagonal.Sigmas(np.full(36, 1e-3))
state = kf.init(mean, np.eye(36) * 1e-2)
state2 = kf.predict(state, F, B, u, Q)
print("Test 4 predict ok")
mean2 = np.array(state2.mean()).reshape(-1, 1)
P2 = np.asarray(state2.covariance())
print(f"  Post-predict mean finite={np.all(np.isfinite(mean2))}, P finite={np.all(np.isfinite(P2))}")
print(f"  P eig range: [{np.linalg.eigvalsh(P2).min():.3e}, {np.linalg.eigvalsh(P2).max():.3e}]")

# Test 5: fresh KF after corrupted P scenario
# Simulate large P (covariance blow-up scenario)
P_big = np.eye(36) * 1e12  # beyond max_cov_eig
from ekf_alt import _make_pd_bounded
P_safe = _make_pd_bounded(P_big)
print(f"\nTest 5: P_big clipped eig range: [{np.linalg.eigvalsh(P_safe).min():.3e}, {np.linalg.eigvalsh(P_safe).max():.3e}]")
kf2 = gtsam.KalmanFilter(36)
s5 = kf2.init(mean, P_safe)
print("Test 5 fresh KF with clipped P ok")
print("mean:", np.array(s5.mean())[:3])

import numpy as np

def group_hrv_synchrony(hrv_matrix):
    """Group HRV Synchrony (ρ_RMSSD) - Chapter 7 metric"""
    if len(hrv_matrix) < 2:
        return 0.0
    corr_matrix = np.corrcoef(hrv_matrix)
    return np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])


def decoupling_velocity_stability(di_velocities):
    """Decoupling-Index Velocity Stability (σ_V_DI)"""
    return np.std(di_velocities) if len(di_velocities) > 0 else 0.0


print("✅ Coherence metrics module created successfully")

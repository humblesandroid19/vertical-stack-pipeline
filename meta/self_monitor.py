import pandas as pd
import numpy as np
from datetime import datetime

class VerticalStackMetaMonitor:
    """Self-monitoring meta-layer (∞-horizon).
    Tracks the Stack's own model drift, coherence, and Decoupling Index over time.
    """

    def __init__(self):
        self.history = []

    def log_coherence(self, decoupling_index_mean, decoupling_velocity_std):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "decoupling_index_mean": float(decoupling_index_mean),
            "decoupling_velocity_std": float(decoupling_velocity_std),
            "overall_coherence_score": float(1.0 / (1.0 + decoupling_velocity_std))
        }
        self.history.append(entry)

        print(f"✅ Meta-monitor logged at {entry['timestamp']}")
        print(f"   Decoupling Index Mean: {entry['decoupling_index_mean']:.3f}")
        print(f"   Velocity Stability:    {entry['decoupling_velocity_std']:.3f}")
        print(f"   Coherence Score:       {entry['overall_coherence_score']:.3f}")
        return entry


if __name__ == "__main__":
    monitor = VerticalStackMetaMonitor()
    monitor.log_coherence(3.67, 0.45)   # baseline example
    monitor.log_coherence(7.99, 1.15)   # strain example

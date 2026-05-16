from pipeline.PhysiologicalRiskClassificationPipeline import PhysiologicalRiskClassificationPipeline
from sklearn.ensemble import RandomForestClassifier
from meta.self_monitor import VerticalStackMetaMonitor
import pandas as pd
import numpy as np

print("🚀 Running full Vertical Stack v3.1...")

# 1. Demo pipeline run
dummy_model = RandomForestClassifier(n_estimators=5, random_state=42).fit(
    np.array([[100, 70, 1.4, 0.0, 0.0], [300, 40, 7.5, 0.2, 1.5]]), np.array([0, 1])
)
pipeline = PhysiologicalRiskClassificationPipeline(trained_model=dummy_model)

payload = {
    "timestamp": pd.date_range('2026-05-16 08:00:00', periods=3, freq='5min'),
    "peak_glucose": [220, 310, 350],
    "min_hrv": [62, 48, 28]
}

result = pipeline.preprocess_and_score(payload)
print("\n📊 Pipeline Result:")
print(result)

# 2. Self-monitoring meta-layer
monitor = VerticalStackMetaMonitor()
monitor.log_coherence(result["decoupling_index"].mean(), result["decoupling_velocity"].std())

print("\n✅ Full Vertical Stack ran successfully!")
print("   The system is now self-monitoring its own coherence.")

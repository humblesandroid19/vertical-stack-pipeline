import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pipeline.PhysiologicalRiskClassificationPipeline import PhysiologicalRiskClassificationPipeline

def test_pipeline_runtime_safety():
    X = np.array([[100, 70, 1.4, 0.0, 0.0], [300, 40, 7.5, 0.2, 1.5]])
    y = np.array([0, 1])
    dummy_model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    pipeline = PhysiologicalRiskClassificationPipeline(trained_model=dummy_model)

    critical_payload = {
        "timestamp": pd.date_range('2026-05-16 08:00:00', periods=3, freq='5min'),
        "peak_glucose": [350.0, 360.0, 380.0],
        "min_hrv": [20.0, 15.0, 10.0]
    }

    scored_df = pipeline.preprocess_and_score(critical_payload, window_size=3)
    assert "TRIGGER_EMERGENCY_SHUTDOWN" in scored_df["hardware_relay_signal"].values
    assert "RESEARCH_REVIEW_REQUIRED" in scored_df["predicted_scenario"].values

Add automated safety test

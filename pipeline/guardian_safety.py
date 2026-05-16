import pandas as pd
import numpy as np

class GuardianSafetyModule:
    """
    Isolated supervisory safety layer. Enforces strict deterministic constraints
    over telemetry input integrity and extreme physiological states independently
    of machine learning classification layers.
    """
    def __init__(self, 
                 missing_threshold: float = 0.40,
                 min_hrv_critical: float = 35.0,
                 decoupling_critical: float = 9.0):
        self.missing_threshold = missing_threshold
        self.min_hrv_critical = min_hrv_critical
        self.decoupling_critical = decoupling_critical

    def evaluate(self, df: pd.DataFrame, null_imputation_burden: float) -> pd.DataFrame:
        df = df.copy()
        df["hardware_relay_signal"] = "EXECUTE_AUTONOMOUS_LOOP"

        for idx, row in df.iterrows():
            if null_imputation_burden > self.missing_threshold:
                df.at[idx, "predicted_scenario"] = "LOW_CONFIDENCE_REVIEW"
                df.at[idx, "hardware_relay_signal"] = "HOLD_CURRENT_INFUSION"
            elif row["min_hrv"] < self.min_hrv_critical or row.get("decoupling_index", 0) >= self.decoupling_critical:
                df.at[idx, "predicted_scenario"] = "RESEARCH_REVIEW_REQUIRED"
                df.at[idx, "hardware_relay_signal"] = "TRIGGER_EMERGENCY_SHUTDOWN"

        return df

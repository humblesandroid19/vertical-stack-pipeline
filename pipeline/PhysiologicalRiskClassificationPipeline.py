import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .guardian_safety import GuardianSafetyModule

class PhysiologicalRiskClassificationPipeline:
    """
    Hardened production-ready data engineering pipeline for the Vertical Stack.
    Research prototype only - strictly non-clinical.
    """
    def __init__(self, trained_model=None, safety_module=None):
        self.model = trained_model
        self.guardian = safety_module or GuardianSafetyModule()

    def preprocess_and_score(self, raw_data_dict: dict, window_size: int = 6) -> pd.DataFrame:
        df = pd.DataFrame(raw_data_dict)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')

        null_imputation_burden = df["peak_glucose"].isna().sum() / len(df) if len(df) > 0 else 1.0

        df = df.ffill().bfill()

        df["min_hrv_safe"] = df["min_hrv"].replace(0, np.nan)
        df["decoupling_index"] = df["peak_glucose"] / df["min_hrv_safe"]
        df["decoupling_index"] = df["decoupling_index"].replace([np.inf, -np.inf], np.nan)
        median_val = df["decoupling_index"].median()
        if pd.isna(median_val):
            median_val = 1.0
        df["decoupling_index"] = df["decoupling_index"].fillna(median_val)

        df["glucose_cv"] = (df["peak_glucose"].rolling(window=window_size, min_periods=3).std() /
                            df["peak_glucose"].rolling(window=window_size, min_periods=3).mean()).fillna(0.0)

        time_diffs = df.index.to_series().diff().dt.total_seconds() / 60.0
        df["decoupling_velocity"] = (df["decoupling_index"].diff() / time_diffs).fillna(0.0)

        feature_cols = ["peak_glucose", "min_hrv", "decoupling_index", "glucose_cv", "decoupling_velocity"]
        features = df[feature_cols]

        df["predicted_scenario"] = self.model.predict(features)
        if df["predicted_scenario"].dtype != object:
            df["predicted_scenario"] = df["predicted_scenario"].astype(object)

        df = self.guardian.evaluate(df, null_imputation_burden)

        return df[["predicted_scenario", "decoupling_index", "decoupling_velocity", "hardware_relay_signal"]].round(2)


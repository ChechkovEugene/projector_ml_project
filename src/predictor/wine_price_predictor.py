import os
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class WinePricePredictor:
    def __init__(self, model: Pipeline, scaler: StandardScaler) -> None:
        self._model = model
        self._scaler = scaler

    @classmethod
    def load(cls, model_folder: str) -> "WinePricePredictor":
        model = joblib.load(os.path.join(model_folder, "model.pkl"))
        scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))
        return cls(model=model, scaler = scaler)

    def predict(self, samples: List[Dict]) -> List[str]:
        samples_df = pd.DataFrame.from_dict(samples)
        feature_names = self._model["reg"].get_booster().feature_names

        numerical = samples_df.select_dtypes(exclude=['object']).columns

        samples_df['Grape'] = samples_df['Grape'].str.split(',')

        def split_list(row):
            return pd.Series(row['Grape'])

        new_df = samples_df.apply(split_list, axis=1).rename(columns=lambda x: f"Grape_{x+1}")
        samples_df = pd.concat([samples_df, new_df], axis=1)
        samples_df.drop('Grape', axis=1, inplace=True)

        category = samples_df.select_dtypes(include=['object']).columns
        samples_df[category] = samples_df[category].astype('category')
        samples_df = pd.get_dummies(samples_df, columns=category)
        samples_df = samples_df.reindex(columns = feature_names, fill_value=False)

        samples_df[numerical] = self._scaler.transform(samples_df[numerical])
        predictions = list(np.exp(self._model.predict(samples_df)))

        return predictions
        
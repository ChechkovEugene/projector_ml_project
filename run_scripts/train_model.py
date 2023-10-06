import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from preprocessor.data_preprocessor import DataPreprocessor
from utils import print_metrics
from sklearn.feature_extraction.text import TfidfVectorizer

import xgboost as xgb

DATA_FOLDER = "data"
PREPROCESSED_DATA_FOLDER = "preprocessed_data"
MODEL_FOLDER = "model"

target = ['Price']

numerical = [
    'Alcohol', 'Sweet', 'Year', 'Volume'
]

categorical = [
    'Producer',
    'Region', 'Country', 'Color', 'Type',
    'Style', 'Sweetness', 'Classification_Category'
]

def main() -> None:

    with open('config.json', 'r') as f:
        parameters = json.load(f)

    data_preprocessor = DataPreprocessor()
    df = pd.read_csv(os.path.join(DATA_FOLDER, "wines.csv"), index_col=0)
    df = data_preprocessor.preprocess_data(df)

    df['Price'] = (df['Price']).apply(np.log)
    grape_count = parameters['grape_count'] \
        if parameters and 'grape_count' in parameters else 3

    for (columnName, _) in df.items():
        if columnName.startswith('Grape_') and (int(columnName.split('_')[1]) <= grape_count):
            categorical.append(columnName)

    add_text_feature = parameters['add_text_feature'] if parameters \
        and 'add_text_feature' in parameters else None

    if add_text_feature:
        df_selected = df[numerical+target+categorical + ['Text_Description']]
    else:
        df_selected = df[numerical+target+categorical]

    df_selected = pd.get_dummies(df_selected, columns=categorical, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df_selected.drop(['Price'], axis=1),
        df_selected['Price'],
        test_size=0.25,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled_numeric = scaler.fit_transform(X_train[numerical])
    X_test_scaled_numeric = scaler.transform(X_test[numerical])

    X_train[numerical] = X_train_scaled_numeric
    X_test[numerical] = X_test_scaled_numeric

    if add_text_feature:
        vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        X_train = vectorizer.fit_transform(X_train["Text_Description"])
        # transformed_texts_df = pd.DataFrame(transformed_texts.todense(), columns=vectorizer.get_feature_names_out())
        # pd.concat([X_train, transformed_texts_df], axis=1)

        X_test = vectorizer.transform(X_test["Text_Description"])
        # transformed_texts_df_test = pd.DataFrame(transformed_texts_test.todense(), columns=vectorizer.get_feature_names_out())
        # X_train = pd.concat([X_train, transformed_texts_df], axis=1)
        # X_test = pd.concat([X_test, transformed_texts_df_test], axis=1)


    # X_train.drop('Text_Description', axis=1, inplace=True)
    # X_test.drop('Text_Description', axis=1, inplace=True)

    model_parameters = parameters['model'] if parameters and 'model' in parameters else None 
    
    xgb_model = xgb.XGBRegressor(**model_parameters, random_state=42) \
        if model_parameters else xgb.XGBRegressor(random_state=42)
    
    xgb_model.fit(X_train, y_train)
    # pipeline = Pipeline()
    
    # if add_text_feature:
    #     text_preprocessing = Pipeline(
    #     [
    #         ('tfidf', TfidfVectorizer(stop_words='english'))       
    #     ]
    #     )

    #     pipeline.steps.append(
    #         [make_column_transformer((text_preprocessing, "Text_Description"),)]
    #     )

    pipeline = Pipeline(
        [("reg", xgb_model)]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print(print_metrics(y_test, predictions))

    joblib.dump(pipeline, os.path.join(MODEL_FOLDER, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_FOLDER, "scaler.pkl"))


if __name__ == "__main__":
    main()

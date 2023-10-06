import os
import pandas as pd
import numpy as np

DATA_FOLDER = "data"

COLUMN_NAMES = ['Назва','Виробник','Артикул','Ціна','Температура подачі','Сорти винограду',
                'Технологія виробництва','Об `єм','Рік','Бренд','Регіон','Країна',
                'Солодкість','Тип напою', 'Колір вина','Склад землі','З чим подавати','Класифікація',
                'Розміщення виноградників','Витримка','Склад винограду','Цукор',
                'Алкоголь, %','Збір урожаю','Розширений колір вина','Аромат','Смак','Цікаве',
                'Стиль вин','Потенціал','Дегустації']

COLUMN_NAMES_ENGLISH = ['Title','Producer','Vendor_code','Price','Temperature','Grape',
                'Technology','Volume','Year','Brand','Region','Country','Sweetness',
                'Type','Color','Soil','Serve_with','Classification','Vineyards_placement',
                'Endurance','Grapes_composition','Sweet','Alcohol','Harvest','Additional_color',
                'Aroma','Taste','Interesting','Style','Potential','Degustations']

BEST_CATEGORIES = ["PDO", "DOP", "AOP", "AOC", "VDQS", "DOCG", "DOC", "DO", "QmP", 
                   "P.D.O", "D.O.P", "A.O.P", "A.O.C", "V.D.Q.S", "D.O.C.G", "D.O.C", "D.O", "Q.m.P",
                   "Denominacion de Origen Calificada", "Appellation", "Denominazione di Origine Controllata",
                   "Denominacao", "Denominazione", "Аppellation", "Appelalation", "Denominacion",
                   "Denomination", "Quali", "Origin", "Origen", "VSQ"]

REGION_CATEGORIES = ["PGI", "IGP", "VdP", "VdT", "IGT", "VDT", "IPR", "Vinho Reginal", "QBA",
                     "P.G.I", "I.G.P", "V.d.P", "V.d.T", "I.G.T", "V.D.T", "I.P.R", "Q.B.A",
                     "Indicazione Geografica Tipica", "Indicazione Geografica Protette",
                     "Indication Geographique Protegee", "Geografica", "Regional", "Pays", "Vino de la Tierra"]

class DataPreprocessor:

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=dict(zip(COLUMN_NAMES, COLUMN_NAMES_ENGLISH)))
        df = self.clear_data(df)
        df.dropna(subset=['Price'], inplace=True)
        df = self.fill_na(df)
        df = self.process_classification(df)
        df = self.process_categorical_columns(df)
        df = self.process_grape_column(df)
        df = self.add_text_column(df)
        return df

    def clear_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Vendor_code'] = df['Vendor_code'].str.replace(r'<[^<>]*>', '', regex=True)
        df['Volume'] = df['Volume'].str.split(' ').str[0]
        df['Sweet'] = df['Sweet'].str.split(' ').str[0]
        df['Sweet'] = df['Sweet'].str.replace(',', '.')
        df['Sweet'] = pd.to_numeric(df['Sweet'], errors='coerce')
        df['Price'] = df['Price'].str.split(' ').str[0]
        df['Temperature'] = df['Temperature'].str.replace('°C', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Volume'] = df['Volume'].str.replace(',', '.')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        return df

    def fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Year'].fillna(df['Year'].mode()[0], inplace=True)
        df['Sweet'].fillna(df['Sweet'].mode()[0], inplace=True) 
        df['Alcohol'].fillna(df['Alcohol'].mode()[0], inplace=True) 
        df['Volume'].fillna(df['Volume'].mode()[0], inplace=True) 
        df['Year'].fillna(df['Year'].mode()[0], inplace=True)
        return df

    def process_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Producer"].fillna("NA", inplace=True)
        df['Producer'] = df['Producer'].astype('category')
        df["Vendor_code"].fillna("NA", inplace=True)
        df['Vendor_code'] = df['Vendor_code'].astype('category')
        df["Brand"].fillna("NA", inplace=True)
        df['Brand'] = df['Brand'].astype('category')
        df["Region"].fillna("NA", inplace=True)
        df['Region'] = df['Region'].astype('category')
        df["Country"].fillna("NA", inplace=True)
        df['Country'] = df['Country'].astype('category')
        df["Type"].fillna("NA", inplace=True)
        df['Type'] = df['Type'].astype('category')
        df["Color"].fillna("NA", inplace=True)
        df['Color'] = df['Color'].astype('category')
        df["Style"].fillna("NA", inplace=True)
        df['Style'] = df['Style'].astype('category')
        df["Sweetness"].fillna("NA", inplace=True)
        df['Sweetness'] = df['Sweetness'].astype('category')
        df["Classification_Category"].fillna("NA", inplace=True)
        df['Classification_Category'] = df['Classification_Category'].astype('category')
        return df

    def process_grape_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Grape'] = df['Grape'].str.split(',')

        def split_list(row):
            return pd.Series(row['Grape'])

        new_df = df.apply(split_list, axis=1).rename(columns=lambda x: f"Grape_{x+1}")
        return pd.concat([df, new_df], axis=1)
    
    def add_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Text_Description'] = df[['Technology', 
                             'Soil', 
                             'Serve_with',
                             'Classification',
                             'Vineyards_placement',
                             'Endurance',
                             'Harvest',
                             'Additional_color',
                             'Aroma',
                             'Taste',
                             'Interesting',
                             'Potential',
                             'Degustations']].apply(lambda x: ','.join(x.dropna()), axis=1)
        return df
    
    def get_category_by_classification(self, classification):
        if any(cat.lower() in str(classification).lower() for cat in BEST_CATEGORIES):
            return "Best"
        if any(cat.lower() in str(classification).lower() for cat in REGION_CATEGORIES):
            return "Region"
        if classification is not np.NaN:
            return "Ordinary"
        else:
            return classification
        
    def process_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Classification_Category'] = df['Classification'].apply(self.get_category_by_classification)
        return df
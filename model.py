from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#Importing data
data = pd.read_csv("data/raw/housing.csv")


numerical_cols = data.select_dtypes(include = 'number').columns
categorical_cols = data.select_dtypes(include = 'str').columns

print(categorical_cols)

#creating pipelines
numerical_pipeline = Pipeline([])
categorical_pipeline = Pipeline([])


preprocessor = ColumnTransformer(
    transformers = [

    ]
)
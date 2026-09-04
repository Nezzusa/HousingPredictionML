import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=0.995):
        self.factor = factor
        self.upper_limits_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.upper_limits_ = X_df.quantile(self.factor).to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, limit in self.upper_limits_.items():
            X_df[col] = X_df[col].clip(upper=limit)
        return X_df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.max_age_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.max_age_ = X_df["housing_median_age"].max()
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        safe_households = df["households"].replace(0, np.nan)
        safe_rooms = df["total_rooms"].replace(0, np.nan)

        df["population_per_household"] = df["population"] / safe_households
        df["rooms_per_household"] = df["total_rooms"] / safe_households
        df["bedrooms_per_room"] = df["total_bedrooms"] / safe_rooms
        df["is_age_capped"] = (df["housing_median_age"] >= self.max_age_).astype(int)
        return df


class ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clusters=10, random_state=42):
        self.n = clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=clusters, random_state=random_state, n_init=10)

    def fit(self, X, y=None):
        coords = X[["latitude", "longitude"]]
        self.kmeans.fit(coords)
        return self

    def transform(self, X):
        df = X.copy()
        coords = df[["latitude", "longitude"]]
        labels = self.kmeans.predict(coords)

        dummies = pd.get_dummies(labels, prefix="cluster", drop_first=True, dtype=int)

        dummies.index = df.index
        df = pd.concat([df, dummies], axis=1)
        return df.drop(columns=["latitude", "longitude"])


#Box-Cox pipeline
box_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("box_cox", PowerTransformer(method="box-cox", standardize=True)),
])

# Capped pipeline
capping_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("capper", OutlierCapper(factor=0.995)),
])

# one hot pipeline
onehot_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
])

# combining pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("box_total_rooms", box_pipeline, ["total_rooms"]),
        ("rooms_per_household_capped", capping_pipeline, ["rooms_per_household"]),
        ("distance", onehot_pipeline, ["ocean_proximity"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
)

pipeline = Pipeline([
    ("engineer", FeatureEngineer()),
    ("geo_clusters", ClusterTransformer(clusters=10, random_state=42)),
    ("preprocessor", preprocessor)
])


data = pd.read_csv("data/raw/housing.csv")
X = data.drop(columns=["median_house_value"])
Y = data["median_house_value"]

target_transformer = PowerTransformer(method="box-cox", standardize=True)
Y_final = target_transformer.fit_transform(Y.values.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.2, random_state=142)

X_train_processed = pipeline.fit_transform(X_train)

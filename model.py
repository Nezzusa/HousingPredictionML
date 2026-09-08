import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.linear_model import LinearRegression
import joblib


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=0.995):
        self.factor = factor
        self.upper_limits_ = None

    def fit(self, features, target=None):
        # Convert to DataFrame to support both NumPy arrays and Pandas objects
        features_df = pd.DataFrame(features)
        self.upper_limits_ = features_df.quantile(self.factor).to_numpy()
        return self

    def transform(self, features):
        features_df = pd.DataFrame(features).copy()
        # Clip each column against the learned quantile vector
        return np.clip(features_df.to_numpy(), a_min=None, a_max=self.upper_limits_)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.max_age_ = None

    def fit(self, features, target=None):
        features_df = pd.DataFrame(features)
        self.max_age_ = features_df["housing_median_age"].max()
        return self

    def transform(self, features):
        features_df = pd.DataFrame(features).copy()
        safe_households = features_df["households"].replace(0, np.nan)
        safe_rooms = features_df["total_rooms"].replace(0, np.nan)

        features_df["population_per_household"] = features_df["population"] / safe_households
        features_df["rooms_per_household"] = features_df["total_rooms"] / safe_households
        features_df["bedrooms_per_room"] = features_df["total_bedrooms"] / safe_rooms
        features_df["is_age_capped"] = (features_df["housing_median_age"] >= self.max_age_).astype(int)
        return features_df


class ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clusters=10, random_state=42):
        self.n = clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=clusters, random_state=random_state, n_init=10)
        # Fix: Enforce fixed category dimensions to avoid predict() shape mismatch
        self.ohe = OneHotEncoder(categories=[list(range(clusters))], drop="first", sparse_output=False)

    def fit(self, features, target=None):
        coords = features[["latitude", "longitude"]]
        self.kmeans.fit(coords)
        self.ohe.fit(np.arange(self.n).reshape(-1, 1))
        return self

    def transform(self, features):
        features_df = features.copy()
        coords = features_df[["latitude", "longitude"]]
        labels = self.kmeans.predict(coords).reshape(-1, 1)

        dummies = self.ohe.transform(labels)
        dummy_cols = [f"cluster_{i}" for i in range(1, self.n)]
        dummies_df = pd.DataFrame(dummies, columns=dummy_cols, index=features_df.index)

        df = pd.concat([features_df, dummies_df], axis=1)
        return df.drop(columns=["latitude", "longitude"])


# Box-Cox pipeline
box_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("box_cox", PowerTransformer(method="box-cox", standardize=True)),
])

# Capping pipeline
capping_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("capper", OutlierCapper(factor=0.995)),
])

# One-hot encoder pipeline
onehot_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
])


# preprocessing (columns divider)
preprocessor = ColumnTransformer(
    transformers=[
        ("box_total_rooms", box_pipeline, ["total_rooms"]),
        ("rooms_per_household_capped", capping_pipeline, ["rooms_per_household"]),
        ("distance", onehot_pipeline, ["ocean_proximity"]),
    ],
    remainder=SimpleImputer(strategy="median"),
    verbose_feature_names_out=False
)

# Full training pipeline
pipeline = Pipeline([
    ("engineer", FeatureEngineer()),
    ("geo_clusters", ClusterTransformer(clusters=10, random_state=42)),
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# import initially cleaned data
data = pd.read_csv("data/processed/clean_data.csv")

# Split raw target and features
X = data.drop(columns=["median_house_value"])
Y = data["median_house_value"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=142)

# Reshape target data
target_transformer = PowerTransformer(method="box-cox", standardize=True)
y_train_box = target_transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_box = target_transformer.transform(y_test.values.reshape(-1, 1)).ravel()

# fitting model and prediction
pipeline.fit(X_train, y_train_box)
prediction = pipeline.predict(X_test)

# Inverse transform to real units
prediction_dollars = target_transformer.inverse_transform(prediction.reshape(-1, 1)).ravel()
y_test_dollars = y_test.values

# Model evaluation
mae_dollars = mean_absolute_error(y_test_dollars, prediction_dollars)
rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, prediction_dollars))
r2 = r2_score(y_test, prediction_dollars)

print("Model performance (Real Units):")
print(f"MAE:  ${mae_dollars:,.2f}")
print(f"RMSE: ${rmse_dollars:,.2f}")
print(f"R²:    {r2:.4f}")

# Export model and target transformer
joblib.dump(pipeline, "models/pipeline.pkl")
joblib.dump(target_transformer, "models/target_transformer.pkl")
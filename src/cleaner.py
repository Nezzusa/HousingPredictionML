import pandas as pd

raw_data = pd.read_csv("../data/raw/housing.csv")

# removes rows where missing values are above 50% threshold
clean_data = raw_data[raw_data.isna().mean(axis=1) <= 0.50]

# remove rows with wrong or impossible data
clean_data = clean_data[clean_data["median_house_value"].notna()]
clean_data = clean_data[clean_data["median_house_value"] > 0]
clean_data = clean_data[clean_data["median_house_value"] < 500000]

# removes duplicates
clean_data = clean_data.drop_duplicates()

# removes rows where negative values exceed certain treshold (50%)
num_cols = clean_data.select_dtypes(include="number").columns
non_negative_cols = num_cols.drop(["latitude", "longitude"], errors="ignore")
clean_data = clean_data[(clean_data[non_negative_cols] < 0).mean(axis=1) <= 0.50]

#validates number of rooms (and includes NaN values)
valid_rooms = (
    (clean_data["total_bedrooms"] <= clean_data["total_rooms"])
    | clean_data["total_bedrooms"].isna()
    | clean_data["total_rooms"].isna()
)

clean_data = clean_data[valid_rooms]

# validates population and households numbers
valid_households = (clean_data["households"] > 0) | clean_data["households"].isna()
valid_population = (clean_data["population"] > 0) | clean_data["population"].isna()
clean_data = clean_data[valid_households & valid_population]

#unifying text
clean_data = clean_data.copy()
clean_data["ocean_proximity"] = clean_data["ocean_proximity"].str.strip()

#saving data
clean_data = clean_data.reset_index(drop=True)
clean_data.to_csv("../data/processed/clean_data.csv", index=False)
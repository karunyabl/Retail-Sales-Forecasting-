import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ============================
# Load data
# ============================
train = pd.read_csv("data/train.csv", low_memory=False)
test = pd.read_csv("data/test.csv", low_memory=False)
store = pd.read_csv("data/store.csv", low_memory=False)


# ============================
# Merge store info
# ============================
train = train.merge(store, on="Store", how="left")
test = test.merge(store, on="Store", how="left")

# Fix StateHoliday datatype
train["StateHoliday"] = train["StateHoliday"].astype(str)
test["StateHoliday"] = test["StateHoliday"].astype(str)


# ============================
# Convert dates + Feature Engineering
# ============================
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

for df in [train, test]:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek


# ============================
# Filter out zero sales
# ============================
train = train[train["Sales"] > 0]


# ============================
# Handle missing values
# ============================
for df in [train, test]:
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("missing")


# ============================
# Features & target
# ============================
target = "Sales"
ignore = ["Sales", "Customers", "Date"]  # KEEP this

features = [c for c in train.columns if c not in ignore]

categorical_features = [
    "StateHoliday",
    "StoreType",
    "Assortment",
    "PromoInterval"
]

# Convert categorical columns
for col in categorical_features:
    if col in train.columns:
        train[col] = train[col].astype("category")
    if col in test.columns:
        test[col] = test[col].astype("category")


X = train[features]
y = train[target]


# ============================
# Train-test split
# ============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================
# LightGBM dataset
# ============================
lgb_train = lgb.Dataset(
    X_train, y_train, categorical_feature=categorical_features
)

lgb_val = lgb.Dataset(
    X_val, y_val, reference=lgb_train,
    categorical_feature=categorical_features
)


# ============================
# Model parameters
# ============================
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}


# ============================
# Train model
# ============================
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=300,
    callbacks=[early_stopping(20), log_evaluation(50)]
)


# ============================
# Validation
# ============================
y_pred = model.predict(X_val, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"Validation RMSE: {rmse:.2f}")


# ============================
# Save results for Power BI
# ============================
val_results = X_val.copy()

val_results["Actual_Sales"] = y_val.values
val_results["Predicted_Sales"] = y_pred

# Add Date back for visualization
val_results["Date"] = train.loc[X_val.index, "Date"]

# Add error column
val_results["Error"] = abs(val_results["Actual_Sales"] - val_results["Predicted_Sales"])

val_results.to_csv("validation_results.csv", index=False)

print("Validation results saved for Power BI!")


# ============================
# Test predictions
# ============================
X_test = test[features]

test_preds = model.predict(X_test, num_iteration=model.best_iteration)

print("Test Predictions (first 10):")
print(test_preds[:10])

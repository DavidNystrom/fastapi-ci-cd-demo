#!/usr/bin/env python3
"""
Grid search popular multiclass models for US Accidents severity prediction (CSV version).

Models:
- Logistic Regression (multinomial, class_weight='balanced')
- Random Forest (class_weight='balanced_subsample')
- Balanced Random Forest (imblearn)
- XGBoost (optional)

Usage:
  python grid_search_severity_csv.py --data /path/to/US_Accidents.csv --max-rows 500000 --target Severity --out-dir runs/
"""

import argparse
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix, make_scorer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# imblearn for BalancedRandomForest
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError:
    BalancedRandomForestClassifier = None

# optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# -----------------------------
# Utilities
# -----------------------------
def load_csv(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load a CSV file with optional row limit."""
    print(f"Reading CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    if max_rows and len(df) > max_rows:
        print(f"Sampling {max_rows} of {len(df):,} rows for grid search...")
        df = df.sample(n=max_rows, random_state=42)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic time, location, and weather features."""
    for col in ["Start_Time", "End_Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")

    if "Start_Time" in df.columns:
        st = df["Start_Time"]
        df["st_hour"] = st.dt.hour
        df["st_dow"] = st.dt.dayofweek
        df["st_month"] = st.dt.month
        df["st_is_weekend"] = (df["st_dow"] >= 5).astype(int)
        df["st_is_night"] = ((df["st_hour"] <= 5) | (df["st_hour"] >= 20)).astype(int)

    if "State" in df.columns:
        df["state"] = df["State"].astype("category")
    if "County" in df.columns:
        df["county"] = df["County"].astype(str)

    for c in ["Weather_Condition", "Wind_Direction", "Sunrise_Sunset",
              "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight", "Timezone"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    num_candidates = [
        "Start_Lat","Start_Lng","End_Lat","End_Lng","Distance(mi)", "Temperature(F)",
        "Wind_Chill(F)","Humidity(%)","Pressure(in)","Visibility(mi)","Wind_Speed(mph)",
        "Precipitation(in)"
    ]
    for c in num_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clip_high_cardinality(df: pd.DataFrame, col: str, top_k: int = 100):
    """Replace rare categories with '__OTHER__'."""
    if col not in df.columns:
        return
    vc = df[col].value_counts()
    keep = set(vc.head(top_k).index)
    df[col] = df[col].where(df[col].isin(keep), "__OTHER__")


def prepare_xy(df: pd.DataFrame, target: str, max_cat_cardinality: int = 100):
    df = df[df[target].notna()]
    df = df[df[target].isin([1, 2, 3, 4])]

    for c in ["City", "County", "Weather_Condition"]:
        clip_high_cardinality(df, c, top_k=max_cat_cardinality)

    y = df[target].astype(int)
    num_cols, cat_cols = [], []

    for c in ["st_hour","st_dow","st_month","st_is_weekend","st_is_night",
              "Start_Lat","Start_Lng","End_Lat","End_Lng","Distance(mi)",
              "Temperature(F)","Wind_Chill(F)","Humidity(%)","Pressure(in)",
              "Visibility(mi)","Wind_Speed(mph)","Precipitation(in)"]:
        if c in df.columns:
            num_cols.append(c)

    for c in ["state","City","county","Weather_Condition","Wind_Direction",
              "Sunrise_Sunset","Civil_Twilight","Nautical_Twilight","Astronomical_Twilight","Timezone"]:
        if c in df.columns:
            cat_cols.append(c)

    X = df[num_cols + cat_cols].copy()
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return pre


def compute_class_weights(y: pd.Series) -> dict[int, float]:
    vals, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(v): float(total / (len(vals) * c)) for v, c in zip(vals, counts)}


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--target", default="Severity")
    parser.add_argument("--max-rows", type=int, default=400000)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--out-dir", default="runs_csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_csv(args.data, args.max_rows)
    df = basic_feature_engineering(df)
    X, y, num_cols, cat_cols = prepare_xy(df, args.target)
    print(f"Final dataset: {X.shape}, Target distribution:\n{y.value_counts(normalize=True)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pre = build_preprocessor(num_cols, cat_cols)
    f1_macro = make_scorer(f1_score, average="macro")
    bal_acc = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    scoring = {"f1_macro": f1_macro, "bal_acc": bal_acc}
    refit_metric = "f1_macro"

    cw = compute_class_weights(y_train)
    results = {}

    # Logistic Regression
    print("\nTraining Logistic Regression (grid search)...")
    logreg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            multi_class="multinomial", solver="saga",
            class_weight="balanced", max_iter=200, n_jobs=-1))
    ])
    logreg_params = {"clf__C": [0.2, 0.5, 1.0, 2.0]}
    gs_log = GridSearchCV(logreg, logreg_params, cv=cv, scoring=scoring, refit=refit_metric, n_jobs=-1, verbose=1)
    gs_log.fit(X_train, y_train)
    results["logreg"] = gs_log

    # Random Forest
    print("\nTraining Random Forest (grid search)...")
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    ])
    rf_params = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [12, 20, None],
        "clf__min_samples_leaf": [1, 3, 5]
    }
    gs_rf = GridSearchCV(rf, rf_params, cv=cv, scoring=scoring, refit=refit_metric, n_jobs=-1, verbose=1)
    gs_rf.fit(X_train, y_train)
    results["random_forest"] = gs_rf

    # Balanced Random Forest
    if BalancedRandomForestClassifier is not None:
        print("\nTraining Balanced Random Forest (grid search)...")
        brf = Pipeline([
            ("pre", pre),
            ("clf", BalancedRandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        brf_params = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [12, 20, None],
            "clf__min_samples_leaf": [1, 3, 5]
        }
        gs_brf = GridSearchCV(brf, brf_params, cv=cv, scoring=scoring, refit=refit_metric, n_jobs=-1, verbose=1)
        gs_brf.fit(X_train, y_train)
        results["balanced_rf"] = gs_brf
    else:
        print("\nSkipping Balanced Random Forest (imblearn not installed).")

    # XGBoost
    if HAS_XGB:
        print("\nTraining XGBoost (grid search)...")
        sample_weights = y_train.map(cw).astype(float).values
        xgb = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(objective="multi:softmax", num_class=4, eval_metric="mlogloss",
                                  tree_method="hist", n_jobs=-1, random_state=42))
        ])
        xgb_params = {
            "clf__n_estimators": [250, 500],
            "clf__max_depth": [6, 10],
            "clf__learning_rate": [0.05, 0.1]
        }
        gs_xgb = GridSearchCV(xgb, xgb_params, cv=cv, scoring=scoring, refit=refit_metric, n_jobs=-1, verbose=1)
        gs_xgb.fit(X_train, y_train, **{"clf__sample_weight": sample_weights})
        results["xgboost"] = gs_xgb
    else:
        print("\nSkipping XGBoost (not installed).")

    # Summaries
    print("\n===== Results (CV f1_macro) =====")
    for name, gs in results.items():
        print(f"{name:20s}: {gs.best_score_:.4f} | {gs.best_params_}")

    # Evaluate on test
    print("\n===== Test evaluation =====")
    for name, gs in results.items():
        model = gs.best_estimator_
        y_pred = model.predict(X_test)
        f1m = f1_score(y_test, y_pred, average="macro")
        ba = balanced_accuracy_score(y_test, y_pred)
        print(f"\nModel: {name}")
        print(f"Test F1_macro: {f1m:.4f}, Balanced Acc: {ba:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=3))

    # Save summary
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump({k: v.best_params_ for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()

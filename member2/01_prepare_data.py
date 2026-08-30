# ============================================================
# PAIMANA-AI - MEMBER 2
# DATA PREPARATION
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Get the PAIMANA-AI project root automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = PROJECT_ROOT / "data" / "processed"

PREDICTOR_FILE = DATA_FOLDER / "member1_ml_predictors.csv"
TARGET_FILE = DATA_FOLDER / "member1_ml_targets.csv"
OUTPUT_FILE = DATA_FOLDER / "member2_ai_dataset.csv"
# ============================================================
# 2. CHECK FILES
# ============================================================

print("=" * 70)
print("PAIMANA-AI - MEMBER 2 DATA PREPARATION")
print("=" * 70)

print("\nChecking input files...")

if not PREDICTOR_FILE.exists():
    print("\nERROR: Predictor file not found!")
    print("Expected location:")
    print(PREDICTOR_FILE)
    raise FileNotFoundError(PREDICTOR_FILE)

if not TARGET_FILE.exists():
    print("\nERROR: Target file not found!")
    print("Expected location:")
    print(TARGET_FILE)
    raise FileNotFoundError(TARGET_FILE)

print("Predictor file found.")
print("Target file found.")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING MEMBER 1 DATASETS")
print("=" * 70)

X = pd.read_csv(PREDICTOR_FILE)
y = pd.read_csv(TARGET_FILE)

print("\nPredictor shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 4. DISPLAY COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTOR COLUMNS")
print("=" * 70)

for i, column in enumerate(X.columns, start=1):
    print(f"{i}. {column}")


print("\n" + "=" * 70)
print("TARGET COLUMNS")
print("=" * 70)

for i, column in enumerate(y.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# 5. REMOVE DUPLICATE COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("REMOVING DUPLICATES")
print("=" * 70)

X = X.loc[:, ~X.columns.duplicated()]
y = y.loc[:, ~y.columns.duplicated()]

print("Duplicate columns removed.")


# ============================================================
# 6. RESET INDEX
# ============================================================

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


# ============================================================
# 7. CHECK ROW COUNT
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ROWS")
print("=" * 70)

print("Predictor rows:", len(X))
print("Target rows:", len(y))

if len(X) != len(y):
    raise ValueError(
        f"\nERROR: Predictor and target row counts do not match!\n"
        f"Predictor rows = {len(X)}\n"
        f"Target rows = {len(y)}"
    )

print("Row counts match.")


# ============================================================
# 8. REMOVE TARGET COLUMNS FROM PREDICTORS
# ============================================================

print("\n" + "=" * 70)
print("REMOVING TARGET VARIABLES FROM INPUT FEATURES")
print("=" * 70)

# Any columns already present in the target dataset
# should not also be present in the predictor dataset.
target_columns = list(y.columns)

removed_columns = []

for column in target_columns:
    if column in X.columns:
        X = X.drop(columns=[column])
        removed_columns.append(column)

if removed_columns:
    print("Removed target columns from predictors:")
    for column in removed_columns:
        print(" -", column)
else:
    print("No target columns were duplicated in predictors.")


# ============================================================
# 9. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("HANDLING MISSING VALUES")
print("=" * 70)

# Numeric columns → fill missing values with median
numeric_columns = X.select_dtypes(include="number").columns

for column in numeric_columns:
    if X[column].isna().any():
        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = X[column].fillna(median_value)


# Text/categorical columns → fill missing values with "UNKNOWN"
categorical_columns = X.select_dtypes(exclude="number").columns

for column in categorical_columns:
    if X[column].isna().any():
        X[column] = X[column].fillna("UNKNOWN")


#
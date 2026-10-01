"""Validate the fixed dataset snapshot before running experiments."""

from pathlib import Path
import hashlib

import pandas as pd


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "inverted_table_tennis_rubber_prices_cleaned.csv"
EXPECTED_SHA256 = "52306a7af8522ecca297338bf5d3f059f805b7aab84f89694fa9c281dba7ded7"
EXPECTED_ROWS = 376
EXPECTED_COLUMNS = 13
REQUIRED_COLUMNS = {
    "Rubber Name", "Brand", "Speed", "Spin", "Control", "Gears",
    "Hardness", "Hardness_ESN_numeric", "Tackiness", "Overall",
    "Sponge Thickness(mm)", "Mean_Sponge_Thickness_mm", "Price_EUR",
}


digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
if digest != EXPECTED_SHA256:
    raise ValueError(f"Dataset checksum mismatch: {digest}")

df = pd.read_csv(DATA)
if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
    raise ValueError(f"Unexpected dataset shape: {df.shape}")
if set(df.columns) != REQUIRED_COLUMNS:
    raise ValueError("Dataset columns do not match the documented snapshot")
if df[list(REQUIRED_COLUMNS)].isna().any().any():
    raise ValueError("Unexpected missing value in released dataset")
if (df["Price_EUR"] <= 0).any():
    raise ValueError("Price_EUR must be positive")

print(f"Dataset verified: {len(df)} rows, {df['Brand'].nunique()} brands, SHA-256 {digest}")

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:24:25 2026

@author: DanO'Brien
"""

import pandas as pd
import numpy as np

# --- INPUT / OUTPUT ---
INFILE  = "/mnt/data/Table5_2 (3).xlsx"      # change if needed
OUTFILE = "Table5_2_long.csv"               # change to .xlsx if you prefer

# --- READ ---
# This file has multi-row headers/notes; row 11 (0-indexed) contains the usable column names.
df = pd.read_excel(INFILE, header=11)

# --- STANDARDIZE COLUMN NAMES (by position; robust to messy header text) ---
# Expected layout: [NAICS, EndUse, Total, Elec, Residual, Distillate/Diesel, NatGas, HGL, Coal, Other]
df = df.iloc[:, :10].copy()  # keep first 10 columns (the data table)
df.columns = [
    "NAICS",
    "EndUse",
    "Total",
    "Net_electricity",
    "Residual_fuel_oil",
    "Distillate_and_diesel",
    "Natural_gas",
    "HGL",
    "Coal",
    "Other",
]

# --- CLEAN NAICS + FILL DOWN ---
# NAICS appears only at the top of each block; forward-fill through the block.
df["NAICS"] = df["NAICS"].ffill()

# --- DROP NON-DATA ROWS ---
# Keep only rows that actually have an EndUse label.
df = df[df["EndUse"].notna()].copy()

# --- HANDLE SUPPRESSION / SYMBOLS ---
# Common MECS symbols: "--", "*", "Q" (suppressed/withheld). Convert them to NaN.
symbol_map = {"--": np.nan, "*": np.nan, "Q": np.nan}
for c in df.columns[2:]:
    df[c] = df[c].replace(symbol_map)

# --- WIDE -> LONG ---
df_long = df.melt(
    id_vars=["NAICS", "EndUse"],
    value_vars=df.columns[2:],   # fuels + Total
    var_name="Fuel",
    value_name="Consumption"
)

# --- NUMERIC COERCION + OPTIONAL FILTERING ---
df_long["Consumption"] = pd.to_numeric(df_long["Consumption"], errors="coerce")

# Drop missing values (suppressed/blank)
df_long = df_long.dropna(subset=["Consumption"]).reset_index(drop=True)

# OPTIONAL: drop Total if you only want fuels
# df_long = df_long[df_long["Fuel"] != "Total"].reset_index(drop=True)

# OPTIONAL: extract numeric NAICS code into its own column (works with "311 - 339 ..." etc.)
df_long["NAICS_code"] = df_long["NAICS"].astype(str).str.extract(r"(\d{3})")

# --- SAVE ---
df_long.to_csv(OUTFILE, index=False)
print(f"Saved long-format data to: {OUTFILE}")
print(df_long.head(10))

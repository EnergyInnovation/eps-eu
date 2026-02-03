# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:59:50 2025

@author: DanO'Brien
"""

import numpy as np
import pandas as pd
import pymrio
from pathlib import Path

# Region mapping
EU27 = {
    "AUT","BEL","BGR","CYP","CZE","DEU","DNK","ESP","EST","FIN","FRA",
    "GRC","HRV","HUN","IRL","ITA","LTU","LUX","LVA","MLT","NLD","POL",
    "PRT","ROU","SVK","SVN","SWE"
}

# Sector mapping
sector_map = {
    "A01": "ISIC_01T03",
    "A02": "ISIC_01T03",
    "A03": "ISIC_01T03",
    "B05": "ISIC_05",
    "B06": "ISIC_06",
    "B07": "ISIC_31T33",        # mining lumped into mfg in EU fuel data
    "B08": "ISIC_07T08",
    "B09": "ISIC_09",
    "C10T12": "ISIC_10T12",
    "C13T15": "ISIC_13T15",
    "C16": "ISIC_16",
    "C17_18": "ISIC_17T18",
    "C19": "ISIC_19",
    "C20": "ISIC_20",
    "C21": "ISIC_21",
    "C22": "ISIC_31T33",        # rubber lumped into mfg in EU fuel data
    "C23": "C23_PARENT",          # to be split later into 231/239
    "C24A": "ISIC_241",
    "C24B": "ISIC_242",
    "C25": "ISIC_28",           # manufacturing lumped in EU fuel data
    "C26": "ISIC_28",           # manufacturing lumped in EU fuel data
    "C27": "ISIC_28",           # manufacturing lumped in EU fuel data
    "C28": "ISIC_28",           
    "C29": "ISIC_29",
    "C301": "ISIC_29",          # vehicles lumped into road in EU fuel data
    "C302T309": "ISIC_29",      # vehicles lumped into road in EU fuel data
    "C31T33": "ISIC_31T33",
    "D": "D_PARENT",              # to be split later into 351 / 352_353
    "E": "ISIC_36T39",
    "F": "ISIC_31T33",          # construction lumped into mfg in EU fuel data
    "G": "ISIC_45T47",
    "H49": "ISIC_49T53",
    "H50": "ISIC_49T53",
    "H51": "ISIC_49T53",
    "H52": "ISIC_49T53",
    "H53": "ISIC_49T53",
    "I": "ISIC_55T56",
    "J58T60": "ISIC_58T60",
    "J61": "ISIC_61",
    "J62_63": "ISIC_62T63",
    "K": "ISIC_64T66",
    "L": "ISIC_68",
    "M": "ISIC_69T82",
    "N": "ISIC_69T82",
    "O": "ISIC_84",
    "P": "ISIC_85",
    "Q": "ISIC_86T88",
    "R": "ISIC_90T96",
    "S": "ISIC_90T96",
    "T": "ISIC_97T98",
}


OECD_CSV = Path("/tmp/mrios/OECD/ICIO2025_2020.csv")
# Note: Place the 2025 ICIO CSV at the path above; outdated pymrio can't download it yet.

def leontief_inverse(A_sq: pd.DataFrame) -> pd.DataFrame:
    """Compute Leontief inverse (I - A)^-1 for a square pandas DataFrame."""
    I = np.eye(A_sq.shape[0])
    L = np.linalg.inv(I - A_sq.values)
    return pd.DataFrame(L, index=A_sq.index, columns=A_sq.columns)

def build_eu_row_agg(regions: list[str], eu_set: set[str]) -> np.ndarray:
    """Build a 2xN aggregation matrix mapping regions -> [EU, ROW]."""
    mat = np.zeros((2, len(regions)), dtype=int)
    for j, r in enumerate(regions):
        mat[0 if r in eu_set else 1, j] = 1
    return mat

# -------------------------
# 1) Load OECD ICIO + aggregate to EU/ROW
# -------------------------
mrio = pymrio.parse_oecd(str(OECD_CSV))

regions = list(mrio.regions)
reg_agg = build_eu_row_agg(regions, EU27)

mrio.aggregate(region_agg=reg_agg, region_names=["EU", "ROW"])
mrio.calc_all()  # ensures A is computed

def map_sector(s: str) -> str:
    # robust to any stray quote chars in labels
    s_clean = str(s).replace("'", "").strip()
    return sector_map.get(s_clean, s_clean)

def aggregate_square_matrix_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates a square matrix (e.g., Z or A) with MultiIndex rows/cols (region, sector)
    by mapping sector -> target sector, within each region.
    """
    # map rows
    r0 = df.index.get_level_values(0)
    r1 = df.index.get_level_values(1).map(map_sector)
    df2 = df.copy()
    df2.index = pd.MultiIndex.from_arrays([r0, r1], names=df.index.names)

    # map cols
    c0 = df.columns.get_level_values(0)
    c1 = df.columns.get_level_values(1).map(map_sector)
    df2.columns = pd.MultiIndex.from_arrays([c0, c1], names=df.columns.names)

    # sum duplicates created by mapping
    df2 = df2.groupby(level=[0, 1]).sum()
    df2 = df2.T.groupby(level=[0, 1]).sum().T
    return df2

def aggregate_vector_by_sector(x: pd.Series) -> pd.Series:
    """Aggregates x with MultiIndex (region, sector) using sector_map."""
    r0 = x.index.get_level_values(0)
    r1 = x.index.get_level_values(1).map(map_sector)
    x2 = x.copy()
    x2.index = pd.MultiIndex.from_arrays([r0, r1], names=x.index.names)
    return x2.groupby(level=[0, 1]).sum()

# --- Work from flows: Z and x ---
Z = mrio.Z.copy()
x = mrio.x.copy()

# Aggregate sectors to your ISIC buckets (keeps C23_PARENT and D_PARENT for later splitting)
Z_agg = aggregate_square_matrix_by_sector(Z)
x_agg = aggregate_vector_by_sector(x)

# Harmonize MultiIndex names + types for safe alignment
col_idx = Z_agg.columns

x_agg = x_agg.copy()
x_agg.index = pd.MultiIndex.from_tuples(
    [(str(r), str(s)) for r, s in x_agg.index],
    names=col_idx.names
)

Z_agg.columns = pd.MultiIndex.from_tuples(
    [(str(r), str(s)) for r, s in Z_agg.columns],
    names=col_idx.names
)

x_agg = x_agg.reindex(Z_agg.columns)

# --- Force x_agg to the exact same order as Z_agg columns (no name-based join) ---
x_aligned = x_agg.reindex(Z_agg.columns)

# Convert to numpy and divide column-wise
x_vec = x_aligned.to_numpy(dtype=float)

Z_vals = Z_agg.to_numpy(dtype=float)

# Safe divide: if x==0, set A column to 0
A_vals = np.divide(Z_vals, x_vec, out=np.zeros_like(Z_vals), where=(x_vec != 0))

A_agg = pd.DataFrame(A_vals, index=Z_agg.index, columns=Z_agg.columns)

# Identify EU/ROW rows/cols on the aggregated system
eu_rows = [i for i in A_agg.index if i[0] == "EU"]
eu_cols = [j for j in A_agg.columns if j[0] == "EU"]
row_rows = [i for i in A_agg.index if i[0] == "ROW"]

# Domestic and imports blocks
A_D = A_agg.loc[eu_rows, eu_cols].copy()
A_M = A_agg.loc[row_rows, eu_cols].copy()

# Fold imports into EU sector rows
A_M.index = pd.MultiIndex.from_arrays(
    [["EU"] * len(A_M.index), A_M.index.get_level_values(1)],
    names=A_D.index.names
)
A_M_folded = A_M.groupby(level=[0, 1]).sum().reindex(A_D.index).fillna(0.0)

A_T = A_D + A_M_folded

# Leontief inverses
LEONTFD_EU = leontief_inverse(A_D)
LEONTFT_EU = leontief_inverse(A_T)

LEONTFD_EU.to_csv("LEONTFD_EU.csv")
LEONTFT_EU.to_csv("LEONTFT_EU.csv")

diffL = (LEONTFT_EU - LEONTFD_EU).abs().values
print("Wrote: LEONTFD_EU.csv and LEONTFT_EU.csv (sector-aggregated)")
print("Max abs diff in L:", float(diffL.max()))
print("Mean abs diff in L:", float(diffL.mean()))

# Optional: confirm the parent sectors exist (needed for RAS split)
print("Has C23_PARENT?", ("EU", "C23_PARENT") in A_D.index)
print("Has D_PARENT?", ("EU", "D_PARENT") in A_D.index)
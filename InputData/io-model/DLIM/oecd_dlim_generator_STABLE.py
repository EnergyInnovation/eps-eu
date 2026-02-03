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
    "B07": "ISIC_07T08",
    "B08": "ISIC_07T08",
    "B09": "ISIC_09",
    "C10T12": "ISIC_10T12",
    "C13T15": "ISIC_13T15",
    "C16": "ISIC_16",
    "C17_18": "ISIC_17T18",
    "C19": "ISIC_19",
    "C20": "ISIC_20",
    "C21": "ISIC_21",
    "C22": "ISIC_22",
    "C23": "C23_PARENT",          # to be split later into 231/239
    "C24A": "ISIC_241",
    "C24B": "ISIC_242",
    "C25": "ISIC_25",
    "C26": "ISIC_26",
    "C27": "ISIC_27",
    "C28": "ISIC_28",
    "C29": "ISIC_29",
    "C301": "ISIC_30",
    "C302T309": "ISIC_30",
    "C31T33": "ISIC_31T33",
    "D": "D_PARENT",              # to be split later into 351 / 352_353
    "E": "ISIC_36T39",
    "F": "ISIC_41T43",
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

A = mrio.A.copy()

# Identify EU/ROW rows/cols (MultiIndex: (region, sector))
eu_rows = [i for i in A.index if i[0] == "EU"]
eu_cols = [j for j in A.columns if j[0] == "EU"]
row_rows = [i for i in A.index if i[0] == "ROW"]

# -------------------------
# 2) Build EU domestic and EU total (imports folded)
# -------------------------
A_D = A.loc[eu_rows, eu_cols].copy()          # EU suppliers -> EU users (domestic)
A_M = A.loc[row_rows, eu_cols].copy()         # ROW suppliers -> EU users (imports)

# Fold imports into EU sector rows: (ROW, sector) -> (EU, sector)
A_M.index = pd.MultiIndex.from_arrays(
    [["EU"] * len(A_M.index), A_M.index.get_level_values(1)],
    names=A_D.index.names
)
A_M_folded = A_M.groupby(level=[0, 1]).sum().reindex(A_D.index).fillna(0.0)

A_T = A_D + A_M_folded                         # total EU coefficients (domestic + imports)

# -------------------------
# 3) Leontief inverses
# -------------------------
LEONTFD_EU = leontief_inverse(A_D)             # domestic-only
LEONTFT_EU = leontief_inverse(A_T)             # total (imports baked in)

LEONTFD_EU.to_csv("LEONTFD_EU.csv")
LEONTFT_EU.to_csv("LEONTFT_EU.csv")

# -------------------------
# 4) Diagnostics
# -------------------------
diffL = (LEONTFT_EU - LEONTFD_EU).abs().values
print("Wrote: LEONTFD_EU.csv and LEONTFT_EU.csv")
print("Max |A_M| (ROW->EU):", float(np.abs(A_M.values).max()))
print("Max |A_T - A_D|:", float(np.abs((A_T - A_D).values).max()))
print("Max abs diff in L:", float(diffL.max()))
print("Mean abs diff in L:", float(diffL.mean()))

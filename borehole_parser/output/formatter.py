"""
formatter.py
-------------
Writes the extracted DataFrame + metadata to various output formats:
  • CSV
  • Excel (.xlsx)  – with the metadata on a separate sheet
  • JSON
  • Parquet

The output directory is created automatically if it does not exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


_SUPPORTED_FORMATS = ("csv", "excel", "json", "parquet")


def save(
    df: pd.DataFrame,
    metadata: Optional[Dict],
    output_path: str,
    fmt: str = "csv",
) -> str:
    """
    Save the extracted data to disk.

    Parameters
    ----------
    df : pd.DataFrame  – extracted borehole data
    metadata : dict    – borehole-level metadata (may be None)
    output_path : str  – path to output file (extension added if missing)
    fmt : str          – one of "csv", "excel", "json", "parquet"

    Returns
    -------
    str  – absolute path to the written file.
    """
    fmt = fmt.lower().strip()
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose from {_SUPPORTED_FORMATS}.")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Add extension if not present
    ext_map = {"csv": ".csv", "excel": ".xlsx", "json": ".json", "parquet": ".parquet"}
    if p.suffix.lower() not in ext_map.values():
        p = p.with_suffix(ext_map[fmt])

    if fmt == "csv":
        df.to_csv(p, index=False)

    elif fmt == "excel":
        with pd.ExcelWriter(str(p), engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Borehole_Data", index=False)
            if metadata:
                meta_df = pd.DataFrame(
                    [(k, v) for k, v in metadata.items() if v is not None],
                    columns=["Field", "Value"],
                )
                meta_df.to_excel(writer, sheet_name="Metadata", index=False)

    elif fmt == "json":
        out: Dict = {"data": df.to_dict(orient="records")}
        if metadata:
            out["metadata"] = {k: v for k, v in metadata.items() if v is not None}
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str, ensure_ascii=False)

    elif fmt == "parquet":
        # Cast all object columns to string for parquet compatibility
        df_out = df.copy()
        for col in df_out.select_dtypes(include="object").columns:
            df_out[col] = df_out[col].astype(str)
        df_out.to_parquet(p, index=False)

    return str(p.resolve())


def print_summary(df: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
    """Print a human-readable extraction summary to stdout."""
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    if metadata:
        for key, val in metadata.items():
            if val:
                label = key.replace("_", " ").upper()
                print(f"  {label:<30s}: {val}")
        print()

    print(f"  Rows extracted   : {len(df)}")
    print(f"  Columns detected : {len(df.columns)}")
    print(f"  Column names     : {', '.join(df.columns.tolist())}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print(f"\n  Numeric columns  : {', '.join(numeric_cols)}")

    if "depth_m" in df.columns or "depth_from_m" in df.columns:
        depth_col = "depth_m" if "depth_m" in df.columns else "depth_from_m"
        d = pd.to_numeric(df[depth_col], errors="coerce").dropna()
        if len(d):
            print(f"  Depth range (m)  : {d.min():.1f} – {d.max():.1f}")

    if "spt_n_value" in df.columns:
        n = pd.to_numeric(df["spt_n_value"], errors="coerce").dropna()
        if len(n):
            print(f"  SPT N range      : {int(n.min())} – {int(n.max())}")

    print("=" * 60 + "\n")

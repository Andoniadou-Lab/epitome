import os
import time
from pathlib import Path

import pandas as pd


def detect_delimiter(file_path):
    """
    Detect the delimiter in a text file by checking the first non-empty line.
    Returns the most likely delimiter from common options.
    """
    delimiters = ["\t", ",", ";", "|", " "]
    with open(file_path, "r") as f:
        line = ""
        while not line:
            line = f.readline()
            if line == "":
                break
            line = line.strip()

        if not line:
            return "\t"

        counts = {d: line.count(d) for d in delimiters}
        max_delimiter = max(counts.items(), key=lambda x: x[1])

        if max_delimiter[1] == 0:
            if " " in line:
                return " "
            return "\t"
        return max_delimiter[0]


def find_files(base_path):
    """
    Recursively find all CSV, TSV, and TXT files in base_path and all subdirectories.
    Returns a list of Path objects.
    """
    skip_dir_names = {".git", "__pycache__", "analytics", "node_modules"}
    all_files = []
    base_path = Path(base_path)
    for path in base_path.rglob("*"):
        if path.suffix.lower() not in [".csv", ".tsv", ".txt"]:
            continue
        if path.name.startswith("."):
            continue
        if any(part in skip_dir_names for part in path.parts):
            continue
        all_files.append(path)
    return all_files


_SINGLE_COL_KW = dict(
    dtype=str, engine="python", encoding="utf-8", encoding_errors="replace"
)
_MULTI_COL_KW = dict(engine="c", encoding="utf-8", encoding_errors="replace")


def read_file_with_smart_header(file_path, delimiter):
    """
    Read file with smart header detection:
    - If single column: always no header (index lists like genes/rows), keep as str
    - If multiple columns: assume header row and let pandas infer numeric dtypes
    """
    try:
        # Peek without treating the first line as a header — critical for
        # single-column gene/row index TSVs (otherwise the first entry is lost).
        df_peek = pd.read_csv(
            file_path,
            delimiter=delimiter,
            header=None,
            nrows=5,
            **_SINGLE_COL_KW,
        )
        num_columns = df_peek.shape[1]

        if num_columns <= 1:
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                header=None,
                **_SINGLE_COL_KW,
            )
            print("  Single column detected - reading without header (str)")
        else:
            # Infer dtypes — forcing str breaks numeric overview counts / p-values.
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, **_MULTI_COL_KW)
            except Exception:
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    engine="python",
                    encoding="utf-8",
                    encoding_errors="replace",
                )
            print(f"  {num_columns} columns detected - reading with header (inferred dtypes)")

        return df
    except pd.errors.EmptyDataError:
        print("  Empty file detected")
        return pd.DataFrame()


def convert_to_parquet(
    base_path,
    delete_original=False,
    force=False,
    interactive=True,
):
    """
    Convert all CSV, TSV, and TXT files to Parquet format.

    Args:
        base_path: Directory to start searching from
        delete_original: If True, deletes original files after successful conversion
        force: If True, overwrite existing parquet files
        interactive: If False, skip the confirmation prompt
    """
    all_files = find_files(base_path)
    if not force:
        all_files = [f for f in all_files if not f.with_suffix(".parquet").exists()]

    total_files = len(all_files)

    print(f"Found {total_files} files to convert" + (" (force overwrite)" if force else ""))
    for file in all_files:
        print(f"  - {file}")

    if not total_files:
        print("No CSV, TSV, or TXT files need conversion!")
        return {"success": 0, "errors": 0, "total": 0}

    if interactive:
        confirm = input(f"\nProceed with converting {total_files} files? (y/n): ")
        if confirm.lower() != "y":
            print("Operation cancelled")
            return {"success": 0, "errors": 0, "total": total_files, "cancelled": True}

    start_time = time.time()
    success_count = 0
    error_count = 0
    error_files = []

    for idx, file_path in enumerate(all_files, 1):
        try:
            if file_path.suffix.lower() == ".csv":
                delimiter = ","
            elif file_path.suffix.lower() == ".tsv":
                # Many epitome TSVs are one value per line (no tab). Treat as
                # newline-separated single column via sep that won't split names.
                with open(file_path, "r") as fh:
                    sample = fh.read(4096)
                delimiter = "\t" if "\t" in sample else None
            else:
                delimiter = detect_delimiter(file_path)
                print(f"  Detected delimiter: {delimiter!r}")

            print(f"\nProcessing {idx}/{total_files}: {file_path}")
            if delimiter is None:
                # One token per line (dotplot genes/rows, etc.)
                df = pd.read_csv(file_path, header=None, **_SINGLE_COL_KW)
                print("  Line-list file detected - reading without header (str)")
            else:
                df = read_file_with_smart_header(file_path, delimiter)

            if df.empty:
                print("  Skipping empty file")
                continue

            parquet_path = file_path.with_suffix(".parquet")
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            print(f"✓ Successfully converted to {parquet_path}")
            print(f"  Rows: {len(df):,}  Cols: {df.shape[1]}")
            print(f"  Original size: {os.path.getsize(file_path):,} bytes")
            print(f"  Parquet size: {os.path.getsize(parquet_path):,} bytes")

            if delete_original:
                file_path.unlink()
                print(f"✓ Deleted original file: {file_path}")

            success_count += 1

        except Exception as e:
            print(f"✗ Error converting {file_path}: {str(e)}")
            error_count += 1
            error_files.append((str(file_path), str(e)))

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("Conversion Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {error_count}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

    if error_files:
        print("\nFiles that failed to convert:")
        for file, error in error_files:
            print(f"  - {file}")
            print(f"    Error: {error}")

    return {"success": success_count, "errors": error_count, "total": total_files}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert CSV/TSV/TXT files to Parquet format"
    )
    parser.add_argument("path", help="Directory path to search for files")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete original files after successful conversion",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing parquet files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()
    convert_to_parquet(
        args.path,
        delete_original=args.delete,
        force=args.force,
        interactive=not args.yes,
    )

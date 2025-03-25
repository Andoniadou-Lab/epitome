import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
from pathlib import Path

from config import Config

BASE_PATH = Config.BASE_PATH


def test_load():
    """Test loading each data type and print info about the loaded data"""
    try:
        print("\nTesting individual data loads:")

        # Test genes parquet
        print("\nTesting genes parquet:")
        genes = pd.read_parquet(f"{BASE_PATH}/data/expression/v_0.01/genes.parquet")
        print(f"✓ Genes loaded successfully")
        print(f"Shape: {genes.shape}")
        print(f"Columns: {genes.columns.tolist()}")
        print(f"First few rows:\n{genes.head()}")

        # Test metadata parquet
        print("\nTesting metadata parquet:")
        meta_data = pd.read_parquet(
            f"{BASE_PATH}/data/expression/v_0.01/meta_data.parquet"
        )
        print(f"✓ Metadata loaded successfully")
        print(f"Shape: {meta_data.shape}")
        print(f"Columns: {meta_data.columns.tolist()}")

        # Test matrix
        print("\nTesting matrix:")
        matrix = scipy.io.mmread(
            f"{BASE_PATH}/data/expression/v_0.01/normalized_data.mtx"
        )
        print(f"✓ Matrix loaded successfully")
        print(f"Shape: {matrix.shape}")
        print(f"Type: {type(matrix)}")

        # Test full data load
        print("\nTesting full data load:")
        matrix = scipy.io.mmread(
            f"{BASE_PATH}/data/expression/v_0.01/normalized_data.mtx"
        )
        genes = pd.read_parquet(f"{BASE_PATH}/data/expression/v_0.01/genes.parquet")
        meta_data = pd.read_parquet(
            f"{BASE_PATH}/data/expression/v_0.01/meta_data.parquet"
        )
        meta_data = meta_data[
            [
                "new_cell_type",
                "sample",
                "Age_numeric",
                "sc_sn_atac",
                "Comp_sex",
                "Name",
                "Author",
                "Normal",
                "SRA_ID",
            ]
        ]

        if hasattr(matrix, "todense"):
            matrix = matrix.todense()
        matrix = np.log10(matrix + 1)

        print(f"✓ Full data load successful")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Genes shape: {genes.shape}")
        print(f"Metadata shape: {meta_data.shape}")

        return True

    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting data load test...")
    success = test_load()
    print("\nTest completed.")
    print("Status:", "✓ Success" if success else "✗ Failed")

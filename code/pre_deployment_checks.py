#!/usr/bin/env python3
"""
Pre-Deployment Check Script for Epitome

This script performs multiple pre-deployment checks including:
1. Dataset detection and emptiness validation
2. Stale file detection (files older than 2 weeks)
3. Data loader testing
4. Parquet conversion of CSV/TSV/TXT files
5. Capture package dependencies
"""

import os
import sys
import glob
import time
import subprocess
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# Import config
try:
    from config import Config

    BASE_PATH = Config.BASE_PATH
except ImportError:
    print(
        f"{Fore.RED}Error: Cannot import Config. Please ensure you're running this from the correct directory.{Style.RESET_ALL}"
    )
    sys.exit(1)


def print_header(message):
    """Print a formatted header message"""
    border = "=" * (len(message) + 4)
    print(f"\n{Fore.CYAN}{border}")
    print(f"| {message} |")
    print(f"{border}{Style.RESET_ALL}")


def print_success(message):
    """Print a success message"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_warning(message):
    """Print a warning message"""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")


def print_error(message):
    """Print an error message"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def check_datasets():
    """Check for datasets and validate their contents"""
    print_header("Checking Datasets")

    # Look for version directories directly in data/ subdirectories
    version_dirs = []
    data_path = os.path.join(BASE_PATH, "data")

    # If data path exists, look through its subdirectories
    if os.path.exists(data_path) and os.path.isdir(data_path):
        for data_subdir in os.listdir(data_path):
            data_subdir_path = os.path.join(data_path, data_subdir)
            if os.path.isdir(data_subdir_path):
                for item in os.listdir(data_subdir_path):
                    if item.startswith("v_"):
                        full_path = os.path.join(data_subdir_path, item)
                        if os.path.isdir(full_path):
                            version_dirs.append(full_path)

    # Example path: /Users/k23030440/Library/CloudStorage/OneDrive-King'sCollegeLondon/PhD/Year_two/Aim 1/epitome/data/aging/v_0.01
    if not version_dirs:
        print_warning("No version directories (v_*) found in data subdirectories!")
        print(
            f"Looking for directories like: {os.path.join(BASE_PATH, 'data', '*', 'v_*')}"
        )
        return False

    print(f"Found {len(version_dirs)} version directories across data subdirectories:")
    for vdir in sorted(version_dirs):
        print(f"  - {os.path.relpath(vdir, BASE_PATH)}")

    all_valid = True

    # Get unique version names (e.g., v_0.01)
    unique_versions = set()
    for vdir in version_dirs:
        version_name = os.path.basename(vdir)
        unique_versions.add(version_name)

    # Check for each version if we have all expected data directories
    for version in unique_versions:
        print(f"\nChecking version: {version}")

        # Expected data types for each version
        expected_data_types = [
            "expression",
            "accessibility",
            "chromvar",
            "dotplot",
            "markers",
            "isoforms",
            "cell_proportion",
            "lig_rec",
            "aging",
            "curation",
            "figures",
            "gene_group_annotation",
            "overview",
        ]

        for data_type in expected_data_types:
            expected_path = os.path.join(data_path, data_type, version)

            if not os.path.exists(expected_path):
                print_warning(f"Data type '{data_type}' not found for {version}")
                all_valid = False
            elif not os.listdir(expected_path):
                print_error(f"Directory for '{data_type}' in {version} is empty!")
                all_valid = False
            else:
                print_success(f"Found data for '{data_type}' in {version}")

    if all_valid:
        print_success("All expected data types have content for each version.")
    else:
        print_warning(
            "Some expected data types are missing or empty. Check the warnings above."
        )

    return all_valid


def check_stale_files():
    """Check for files older than 2 weeks, only within v_ directories"""
    print_header("Checking for Stale Files (Older than 2 weeks)")

    # Calculate the timestamp for 2 weeks ago
    two_weeks_ago = datetime.now() - timedelta(weeks=2)
    two_weeks_ago_timestamp = two_weeks_ago.timestamp()

    # Find all version directories within data subdirectories
    data_path = os.path.join(BASE_PATH, "data")
    version_dirs = []

    if os.path.exists(data_path) and os.path.isdir(data_path):
        for data_subdir in os.listdir(data_path):
            data_subdir_path = os.path.join(data_path, data_subdir)
            if os.path.isdir(data_subdir_path):
                for item in os.listdir(data_subdir_path):
                    if item.startswith("v_"):
                        full_path = os.path.join(data_subdir_path, item)
                        if os.path.isdir(full_path):
                            version_dirs.append(full_path)

    if not version_dirs:
        print_warning(f"No version directories (v_*) found in data subdirectories!")
        print(
            f"Looking for directories like: {os.path.join(BASE_PATH, 'data', '*', 'v_*')}"
        )
        return True

    print(f"Found {len(version_dirs)} version directories for stale file checking:")
    for vdir in sorted(version_dirs):
        print(f"  - {os.path.relpath(vdir, BASE_PATH)}")

    # Get all files recursively within version directories
    all_files = []
    for v_dir in version_dirs:
        for root, dirs, files in os.walk(v_dir):
            # Skip hidden directories (starting with .)
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if not file.startswith("."):  # Skip hidden files
                    all_files.append(os.path.join(root, file))

    # Check file age
    old_files = []
    for file_path in all_files:
        try:
            mtime = os.path.getmtime(file_path)
            if mtime < two_weeks_ago_timestamp:
                old_files.append((file_path, datetime.fromtimestamp(mtime)))
        except OSError:
            print_warning(f"Cannot access file: {file_path}")

    # Report old files
    if old_files:
        print_warning(
            f"Found {len(old_files)} files older than 2 weeks in v_* directories:"
        )

        # Group by directory for cleaner output
        by_directory = {}
        for file_path, timestamp in old_files:
            directory = os.path.dirname(file_path)
            if directory not in by_directory:
                by_directory[directory] = []
            by_directory[directory].append((os.path.basename(file_path), timestamp))

        # Print by directory
        for directory, files in by_directory.items():
            print(f"\nDirectory: {directory}")
            for filename, timestamp in files:
                print(
                    f"  - {filename} (Last modified: {timestamp.strftime('%Y-%m-%d')})"
                )
    else:
        print_success("No stale files found in v_* directories.")

    return len(old_files) == 0


def run_test_loader():
    """Run the test_loader.py script"""
    print_header("Running Test Data Loader")

    # Path to test_loader.py
    test_loader_path = os.path.join(BASE_PATH, "code", "modules", "test_loader.py")

    if not os.path.exists(test_loader_path):
        print_error(f"Test loader script not found at {test_loader_path}")
        return False

    try:
        print(f"Running {test_loader_path}...")

        # Method 1: Run as module with importlib
        spec = importlib.util.spec_from_file_location("test_loader", test_loader_path)
        test_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_loader)
        result = test_loader.test_load()

        if result:
            print_success("Test loader executed successfully.")
        else:
            print_error("Test loader reported errors.")

        return result
    except Exception as e:
        print_error(f"Error running test loader: {str(e)}")
        return False


def run_efficient_files_converter():
    """Run the efficient_files.py script to convert files to parquet format"""
    print_header("Running Efficient Files Converter")

    try:
        print("Checking for files that could be converted to Parquet format...")

        # Find all CSV, TSV, and TXT files
        csv_files = list(Path(BASE_PATH).rglob("*.csv"))
        tsv_files = list(Path(BASE_PATH).rglob("*.tsv"))
        txt_files = list(Path(BASE_PATH).rglob("*.txt"))

        # Find corresponding Parquet files
        converted_csv = sum(1 for f in csv_files if f.with_suffix(".parquet").exists())
        converted_tsv = sum(1 for f in tsv_files if f.with_suffix(".parquet").exists())
        converted_txt = sum(1 for f in txt_files if f.with_suffix(".parquet").exists())

        # Report conversion status
        print(f"CSV files: {converted_csv}/{len(csv_files)} converted to Parquet")
        print(f"TSV files: {converted_tsv}/{len(tsv_files)} converted to Parquet")
        print(f"TXT files: {converted_txt}/{len(txt_files)} converted to Parquet")

        total_files = len(csv_files) + len(tsv_files) + len(txt_files)
        total_converted = converted_csv + converted_tsv + converted_txt

        if total_files == total_converted:
            print_success(
                f"All {total_files} files have corresponding Parquet versions"
            )
        else:
            print_warning(
                f"{total_converted}/{total_files} files have been converted to Parquet"
            )
            print_warning(
                f"{total_files - total_converted} files still need conversion"
            )

            # List some files that need conversion as examples
            if converted_csv < len(csv_files):
                needs_conversion = [
                    f for f in csv_files if not f.with_suffix(".parquet").exists()
                ]
                print_warning(
                    f"CSV files needing conversion (showing up to 3 examples):"
                )
                for f in needs_conversion[:3]:
                    print(f"  - {f.relative_to(BASE_PATH)}")

            if converted_tsv < len(tsv_files):
                needs_conversion = [
                    f for f in tsv_files if not f.with_suffix(".parquet").exists()
                ]
                print_warning(
                    f"TSV files needing conversion (showing up to 3 examples):"
                )
                for f in needs_conversion[:3]:
                    print(f"  - {f.relative_to(BASE_PATH)}")

            if converted_txt < len(txt_files):
                needs_conversion = [
                    f for f in txt_files if not f.with_suffix(".parquet").exists()
                ]
                print_warning(
                    f"TXT files needing conversion (showing up to 3 examples):"
                )
                for f in needs_conversion[:3]:
                    print(f"  - {f.relative_to(BASE_PATH)}")

        # Ask if user wants to run conversion
        if total_files > total_converted:
            user_input = (
                input(
                    f"Would you like to convert the remaining {total_files - total_converted} files to Parquet? (y/n): "
                )
                .strip()
                .lower()
            )

            if user_input == "y":
                print("Running conversion process...")

                # Import the converter module
                converter_path = os.path.join(
                    BASE_PATH, "code", "modules", "efficient_files.py"
                )
                if not os.path.exists(converter_path):
                    print_error(
                        f"Efficient files converter script not found at {converter_path}"
                    )
                    return False

                spec = importlib.util.spec_from_file_location(
                    "efficient_files", converter_path
                )
                converter = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(converter)

                # Convert files one by one to avoid overload
                conversion_count = 0

                # Process CSV files
                for file_path in [
                    f for f in csv_files if not f.with_suffix(".parquet").exists()
                ]:
                    try:
                        print(f"Converting {file_path.relative_to(BASE_PATH)}...")
                        # Read the file using pandas
                        df = pd.read_csv(file_path)
                        # Save as parquet
                        df.to_parquet(file_path.with_suffix(".parquet"), index=False)
                        conversion_count += 1
                    except Exception as e:
                        print_error(f"Error converting {file_path}: {str(e)}")

                # Process TSV files
                for file_path in [
                    f for f in tsv_files if not f.with_suffix(".parquet").exists()
                ]:
                    try:
                        print(f"Converting {file_path.relative_to(BASE_PATH)}...")
                        # Read the file using pandas
                        df = pd.read_csv(file_path, sep="\t")
                        # Save as parquet
                        df.to_parquet(file_path.with_suffix(".parquet"), index=False)
                        conversion_count += 1
                    except Exception as e:
                        print_error(f"Error converting {file_path}: {str(e)}")

                # Process TXT files
                for file_path in [
                    f for f in txt_files if not f.with_suffix(".parquet").exists()
                ]:
                    try:
                        print(f"Converting {file_path.relative_to(BASE_PATH)}...")
                        # Detect delimiter
                        with open(file_path, "r") as f:
                            first_line = f.readline().strip()

                        # Simple delimiter detection
                        if "\t" in first_line:
                            delimiter = "\t"
                        elif "," in first_line:
                            delimiter = ","
                        else:
                            delimiter = None  # Let pandas guess

                        # Read the file using pandas
                        if delimiter:
                            df = pd.read_csv(file_path, delimiter=delimiter)
                        else:
                            df = pd.read_csv(file_path, delim_whitespace=True)

                        # Save as parquet
                        df.to_parquet(file_path.with_suffix(".parquet"), index=False)
                        conversion_count += 1
                    except Exception as e:
                        print_error(f"Error converting {file_path}: {str(e)}")

                print_success(
                    f"Successfully converted {conversion_count} files to Parquet format"
                )
            else:
                print("Skipping file conversion.")

        return True
    except Exception as e:
        print_error(f"Error checking file conversion status: {str(e)}")
        return False


def capture_package_versions():
    """Run versions.py to capture current package versions"""
    print_header("Capturing Package Versions")

    # Path to versions.py
    versions_path = os.path.join(BASE_PATH, "code", "versions.py")

    if not os.path.exists(versions_path):
        print_error(f"Versions script not found at {versions_path}")
        return False

    try:
        print(f"Running {versions_path}...")

        # Method 1: Run as module with importlib
        spec = importlib.util.spec_from_file_location("versions", versions_path)
        versions_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(versions_module)
        result = versions_module.save_package_versions()

        if result:
            print_success("Package versions captured successfully.")
        else:
            print_error("Failed to capture package versions.")

        return result
    except Exception as e:
        print_error(f"Error capturing package versions: {str(e)}")
        return False


def check_code_style():
    """Check code style with pycodestyle and fix with Black if requested"""
    print_header("Checking Code Style")

    # Check if pycodestyle is installed
    try:
        subprocess.run(["pycodestyle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning(
            "pycodestyle not installed or not working. Skipping code style check."
        )
        print_warning("Consider installing with: pip install pycodestyle")
        return True

    # Check if black is installed for formatting
    black_available = False
    try:
        subprocess.run(["black", "--version"], check=True, capture_output=True)
        black_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("black not installed. Will not be able to auto-format code.")
        print_warning("Consider installing with: pip install black")

    # Check code style for Python files in the code directory
    code_dir = os.path.join(BASE_PATH, "code")
    if not os.path.exists(code_dir):
        print_warning(f"Code directory not found at {code_dir}")
        return True

    print(f"Checking code style in {code_dir}...")

    try:
        result = subprocess.run(
            ["pycodestyle", "--max-line-length=100", code_dir],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print_success("No code style issues found.")
            return True
        else:
            # Split the output into lines and count them
            style_issues = result.stdout.strip().split("\n")
            total_issues = len(style_issues)

            print_warning(f"Found {total_issues} code style issues.")
            print("First 10 issues:")
            for issue in style_issues[:10]:
                print(f"  {issue}")

            if total_issues > 10:
                print(f"  ... and {total_issues - 10} more issues")

            if black_available:
                fix_issues = (
                    input(
                        "Would you like to use Black to auto-format all Python files? (y/n): "
                    )
                    .strip()
                    .lower()
                )

                if fix_issues == "y":
                    print("Running Black to format Python files...")

                    # Find all Python files recursively
                    python_files = []
                    for root, dirs, files in os.walk(code_dir):
                        python_files.extend(
                            [os.path.join(root, f) for f in files if f.endswith(".py")]
                        )

                    formatted_count = 0
                    for py_file in python_files:
                        try:
                            print(
                                f"Formatting {os.path.relpath(py_file, BASE_PATH)}..."
                            )
                            subprocess.run(
                                ["black", py_file], check=True, capture_output=True
                            )
                            formatted_count += 1
                        except subprocess.CalledProcessError as e:
                            print_error(
                                f"Error formatting {py_file}: {e.stderr.decode('utf-8')}"
                            )

                    print_success(
                        f"Successfully formatted {formatted_count} Python files"
                    )

                    # Run pycodestyle again to check if issues were fixed
                    result = subprocess.run(
                        ["pycodestyle", "--max-line-length=100", code_dir],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        print_success("All code style issues fixed!")
                        return True
                    else:
                        remaining_issues = result.stdout.strip().split("\n")
                        print_warning(
                            f"{len(remaining_issues)} code style issues remain after formatting."
                        )
                        print_warning("Some issues may require manual fixes.")
                        return False
                else:
                    print("Skipping code formatting.")

            return False
    except Exception as e:
        print_error(f"Error checking code style: {str(e)}")
        return False


def check_data_consistency():
    """Check data consistency across different modules"""
    print_header("Checking Data Consistency")

    # This is a more complex check that would require understanding the specific data formats
    # For now, we'll just check that required directories exist for each version

    versions = [
        d
        for d in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, d)) and d.startswith("v_")
    ]

    all_consistent = True
    for version in versions:
        version_path = os.path.join(BASE_PATH, version)

        required_dirs = [
            "expression",
            "accessibility",
            "chromvar",
            "dotplot",
            "markers",
            "isoforms",
            "cell_proportion",
        ]

        for required_dir in required_dirs:
            dir_path = os.path.join(version_path, required_dir)
            if not os.path.exists(dir_path):
                print_error(
                    f"Required directory '{required_dir}' missing for version {version}"
                )
                all_consistent = False
            elif not os.listdir(dir_path):
                print_warning(
                    f"Directory '{required_dir}' is empty for version {version}"
                )

    if all_consistent:
        print_success("All required directories exist for all versions.")

    return all_consistent


def check_sc_data_datasets():
    """Count files in sc_data/datasets directories"""
    print_header("Checking Single-Cell Dataset Files")

    datasets_path = os.path.join(BASE_PATH, "sc_data", "datasets")
    if not os.path.exists(datasets_path):
        print_warning(f"sc_data/datasets directory not found at {datasets_path}")
        return True

    # Find version subdirectories
    version_dirs = [
        d
        for d in os.listdir(datasets_path)
        if os.path.isdir(os.path.join(datasets_path, d)) and d.startswith("v_")
    ]

    if not version_dirs:
        print_warning("No version directories found in sc_data/datasets!")
        return True

    total_h5ad_files = 0
    print(f"Found {len(version_dirs)} version directories in sc_data/datasets")

    for vdir in version_dirs:
        version_path = os.path.join(datasets_path, vdir)
        h5ad_dir = os.path.join(version_path, "epitome_h5_files")

        if not os.path.exists(h5ad_dir):
            print_warning(f"epitome_h5_files directory not found for {vdir}")
            continue

        h5ad_files = [f for f in os.listdir(h5ad_dir) if f.endswith(".h5ad")]
        total_h5ad_files += len(h5ad_files)
        print(f"- {vdir}: {len(h5ad_files)} h5ad files")

    if total_h5ad_files > 0:
        print_success(
            f"Found a total of {total_h5ad_files} h5ad files across all versions"
        )
    else:
        print_warning("No h5ad files found in any version directory!")

    return True

def run_generate_overview_plots():
    """Run the generate_overview_plots.py script"""
    print_header("Running Generate Overview Plots")

    # Path to generate_overview_plots.py
    overview_plots_path = os.path.join(
        BASE_PATH, "code", "modules", "generate_overview_plots.py"
    )

    if not os.path.exists(overview_plots_path):
        print_error(f"Overview plots script not found at {overview_plots_path}")
        return False

    try:
        print(f"Running {overview_plots_path}...")

        # Method 1: Run as module with importlib
        spec = importlib.util.spec_from_file_location(
            "generate_overview_plots", overview_plots_path
        )
        overview_plots_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(overview_plots_module)
        try:
            overview_plots_module.generate_overview_plots(BASE_PATH)
            result = True
        except Exception as e:
            print_error(f"Error running generate_overview_plots: {str(e)}")
            result = False

        if result:
            print_success("Overview plots generated successfully.")
        else:
            print_error("Failed to generate overview plots.")

        return result
    except Exception as e:
        print_error(f"Error running overview plots generator: {str(e)}")
        return False



def main():
    """Run all pre-deployment checks"""
    start_time = time.time()

    print(
        f"{Fore.YELLOW}Starting pre-deployment checks for Epitome at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}"
    )
    print(f"Base path: {BASE_PATH}")

    # Store check results
    results = {}

    # Check datasets
    results["datasets"] = check_datasets()

    # Check sc_data/datasets files
    results["sc_datasets"] = check_sc_data_datasets()

    # Check stale files
    results["stale_files"] = check_stale_files()

    # Ask for file conversion explicitly
    print_header("File Conversion Check")
    run_conversion = (
        input(
            "Would you like to check for and potentially convert files to Parquet format? (y/n): "
        )
        .strip()
        .lower()
    )
    if run_conversion == "y":
        # Run efficient files converter
        results["efficient_files"] = run_efficient_files_converter()
    else:
        print("Skipping file conversion check.")
        results["efficient_files"] = True

    # Run test loader
    results["test_loader"] = run_test_loader()

    # Capture package versions
    results["package_versions"] = capture_package_versions()

    # Check code style
    results["code_style"] = check_code_style()

    # Check data consistency
    results["data_consistency"] = check_data_consistency()

    results["generate_overview_plots"] = run_generate_overview_plots()

    # Print summary
    print_header("Pre-Deployment Check Summary")

    all_passed = True
    for check, result in results.items():
        check_name = check.replace("_", " ").title()
        if result:
            print_success(f"{check_name}: Passed")
        else:
            print_error(f"{check_name}: Failed")
            all_passed = False

    elapsed_time = time.time() - start_time
    print(
        f"\n{Fore.YELLOW}Checks completed in {elapsed_time:.2f} seconds.{Style.RESET_ALL}"
    )

    if all_passed:
        print_success("All pre-deployment checks passed!")
        return 0
    else:
        print_error(
            "Some pre-deployment checks failed. Please review the issues before deploying."
        )
        return 1

    # Print summary
    print_header("Pre-Deployment Check Summary")

    all_passed = True
    for check, result in results.items():
        check_name = check.replace("_", " ").title()
        if result:
            print_success(f"{check_name}: Passed")
        else:
            print_error(f"{check_name}: Failed")
            all_passed = False

    elapsed_time = time.time() - start_time
    print(
        f"\n{Fore.YELLOW}Checks completed in {elapsed_time:.2f} seconds.{Style.RESET_ALL}"
    )

    if all_passed:
        print_success("All pre-deployment checks passed!")
        return 0
    else:
        print_error(
            "Some pre-deployment checks failed. Please review the issues before deploying."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

import os
import pandas as pd
from pathlib import Path
import time

def detect_delimiter(file_path):
    """
    Detect the delimiter in a text file by checking the first non-empty line.
    Returns the most likely delimiter from common options.
    """
    delimiters = ['\t', ',', ';', '|', ' ']
    with open(file_path, 'r') as f:
        # Get first non-empty line
        line = ''
        while not line and line != '':
            line = f.readline().strip()
        
        if not line:
            return '\t'  # default to tab if file is empty
        
        # Count occurrences of each delimiter
        counts = {d: line.count(d) for d in delimiters}
        # Get delimiter with maximum count
        max_delimiter = max(counts.items(), key=lambda x: x[1])
        
        if max_delimiter[1] == 0:
            # If no common delimiter found, check if it's space-separated
            if ' ' in line:
                return ' '
            return '\t'  # default to tab
        return max_delimiter[0]

def find_files(base_path):
    """
    Recursively find all CSV, TSV, and TXT files in base_path and all subdirectories
    Returns a list of Path objects
    """
    all_files = []
    base_path = Path(base_path)
    for path in base_path.rglob('*'):
        if path.suffix.lower() in ['.csv', '.tsv', '.txt']:
            all_files.append(path)
    return all_files

def read_file_with_smart_header(file_path, delimiter):
    """
    Read file with smart header detection:
    - If single column: assume no header
    - If multiple columns: assume header
    """
    try:
        # First read a few lines to check number of columns
        df_peek = pd.read_csv(file_path, delimiter=delimiter, nrows=1)
        num_columns = len(df_peek.columns)
        
        if num_columns == 1:
            # Single column - read without header
            df = pd.read_csv(file_path, delimiter=delimiter, header=None)
            print(f"  Single column detected - reading without header")
        else:
            # Multiple columns - read with header
            df = pd.read_csv(file_path, delimiter=delimiter)
            print(f"  {num_columns} columns detected - reading with header")
        
        return df
    except pd.errors.EmptyDataError:
        print("  Empty file detected")
        return pd.DataFrame()

def convert_to_parquet(base_path, delete_original=False):
    """
    Convert all CSV, TSV, and TXT files to Parquet format.
    Args:
        base_path: Directory to start searching from
        delete_original: If True, deletes original files after successful conversion
    """
    # Find all files
    all_files = find_files(base_path)
    total_files = len(all_files)
    
    print(f"Found {total_files} files to convert")
    print("File list:")
    for file in all_files:
        print(f"  - {file}")
    
    if not total_files:
        print("No CSV, TSV, or TXT files found!")
        return

    # Ask for confirmation
    confirm = input(f"\nProceed with converting {total_files} files? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled")
        return
    
    # Process files
    start_time = time.time()
    success_count = 0
    error_count = 0
    error_files = []
    
    for idx, file_path in enumerate(all_files, 1):
        try:
            # Determine delimiter based on file type and content
            if file_path.suffix.lower() == '.csv':
                delimiter = ','
            elif file_path.suffix.lower() == '.tsv':
                delimiter = '\t'
            else:  # .txt files
                delimiter = detect_delimiter(file_path)
                print(f"  Detected delimiter: '{delimiter}'")
            
            # Read the file with smart header detection
            print(f"\nProcessing {idx}/{total_files}: {file_path}")
            df = read_file_with_smart_header(file_path, delimiter)
            
            if df.empty:
                print("  Skipping empty file")
                continue
            
            # Create output path with same structure but .parquet extension
            parquet_path = file_path.with_suffix('.parquet')
            
            # Create directory if it doesn't exist
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet
            df.to_parquet(parquet_path, index=False)
            print(f"✓ Successfully converted to {parquet_path}")
            print(f"  Original size: {os.path.getsize(file_path):,} bytes")
            print(f"  Parquet size: {os.path.getsize(parquet_path):,} bytes")
            
            # Delete original if requested
            if delete_original:
                file_path.unlink()
                print(f"✓ Deleted original file: {file_path}")
            
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error converting {file_path}: {str(e)}")
            error_count += 1
            error_files.append((str(file_path), str(e)))
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CSV/TSV/TXT files to Parquet format')
    parser.add_argument('path', help='Directory path to search for files')
    parser.add_argument('--delete', action='store_true', 
                       help='Delete original files after successful conversion')
    
    args = parser.parse_args()
    
    # Run conversion
    convert_to_parquet(args.path, args.delete)
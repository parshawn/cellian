"""
Convert large CSV files to Parquet format for faster loading.

This script converts the Perturb-CITE-seq CSV files to Parquet format,
which provides:
- 10-50x faster loading times
- 5-10x smaller file sizes
- Built-in compression
"""

import pandas as pd
import os
import time
from pathlib import Path


def convert_csv_to_parquet(csv_path, parquet_path=None, chunksize=10000):
    """
    Convert a large CSV file to Parquet format.

    Args:
        csv_path: Path to input CSV file
        parquet_path: Path to output Parquet file (default: same as CSV with .parquet)
        chunksize: Number of rows to process at once (for memory efficiency)
    """
    if parquet_path is None:
        parquet_path = csv_path.replace('.csv', '.parquet')

    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    if not csv_path.exists():
        print(f"✗ File not found: {csv_path}")
        return False

    if parquet_path.exists():
        response = input(f"⚠ {parquet_path.name} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("  Skipping...")
            return False

    file_size_gb = csv_path.stat().st_size / (1024**3)
    print(f"\nConverting: {csv_path.name}")
    print(f"  Size: {file_size_gb:.2f} GB")
    print(f"  This may take several minutes...")

    start_time = time.time()

    try:
        # Read CSV with proper index
        print("  Loading CSV...")
        df = pd.read_csv(csv_path, index_col=0)

        print(f"  Shape: {df.shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

        # Convert to Parquet with compression
        print("  Writing Parquet...")
        df.to_parquet(
            parquet_path,
            engine='pyarrow',
            compression='snappy',  # Fast compression
            index=True
        )

        # Get output file size
        parquet_size_gb = parquet_path.stat().st_size / (1024**3)
        compression_ratio = file_size_gb / parquet_size_gb

        elapsed = time.time() - start_time

        print(f"\n✓ Conversion successful!")
        print(f"  Output: {parquet_path}")
        print(f"  Output size: {parquet_size_gb:.2f} GB")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        print(f"  Time taken: {elapsed:.1f} seconds")

        return True

    except MemoryError:
        print(f"\n✗ Memory error! File is too large to load at once.")
        print(f"  Try running on a machine with more RAM (need ~{file_size_gb * 2:.0f}GB)")
        return False

    except Exception as e:
        print(f"\n✗ Error during conversion: {str(e)}")
        return False


def test_loading_speed(parquet_path):
    """
    Test the loading speed of a Parquet file.
    """
    print(f"\nTesting load speed: {parquet_path.name}")

    start_time = time.time()
    df = pd.read_parquet(parquet_path)
    elapsed = time.time() - start_time

    print(f"  ✓ Loaded in {elapsed:.2f} seconds")
    print(f"  Shape: {df.shape}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

    return elapsed


def main():
    print("=" * 70)
    print("CSV TO PARQUET CONVERTER")
    print("=" * 70)

    # Define file paths
    data_dir = Path("/home/nebius/cellian/data/perturb-cite-seq/SCP1064")

    files_to_convert = [
        data_dir / "other" / "RNA_expression.csv",
        data_dir / "expression" / "Protein_expression.csv",
    ]

    converted = []

    for csv_file in files_to_convert:
        if csv_file.exists():
            parquet_file = csv_file.with_suffix('.parquet')
            success = convert_csv_to_parquet(csv_file, parquet_file)

            if success:
                converted.append(parquet_file)
                # Test loading speed
                test_loading_speed(parquet_file)
        else:
            print(f"\n⚠ File not found: {csv_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Converted {len(converted)} file(s):")
    for f in converted:
        print(f"  ✓ {f}")

    if converted:
        print("\nNext steps:")
        print("  1. The hypothesis engine will automatically use .parquet files")
        print("  2. You can safely delete the .csv files to save space")
        print("  3. Re-run your scripts - they should be much faster!")

    print("=" * 70)


if __name__ == "__main__":
    main()

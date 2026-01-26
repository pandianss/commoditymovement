import polars as pl
import os
import sys
import glob
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("parquet_migration", log_file="migration.log")

def convert_to_parquet(source_dir="data/raw", target_dir="data/parquet"):
    """
    Converts all CSV files in source_dir to Parquet in target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(source_dir, "*.csv"))
    
    logger.info(f"Found {len(csv_files)} CSV files to migrate.")
    
    start_total = time.time()
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        name_no_ext = os.path.splitext(filename)[0]
        parquet_file = os.path.join(target_dir, f"{name_no_ext}.parquet")
        
        logger.info(f"Converting {filename}...")
        try:
            # Lazy load CSV -> scan_csv is faster for inference
            # We use eager read_csv here to force load and validate, then write
            # For massive files, we'd use scan_csv().sink_parquet()
            
            # Using partial inference to speed up loading
            start = time.time()
            df = pl.read_csv(csv_file, ignore_errors=True, infer_schema_length=10000)
            
            # Write to Parquet with compression
            df.write_parquet(parquet_file, compression='snappy')
            
            end = time.time()
            original_size = os.path.getsize(csv_file) / (1024 * 1024)
            new_size = os.path.getsize(parquet_file) / (1024 * 1024)
            reduction = (1 - new_size/original_size) * 100
            
            logger.info(f"✓ Converted {filename} in {end-start:.2f}s")
            logger.info(f"  Size: {original_size:.2f}MB -> {new_size:.2f}MB ({reduction:.1f}% reduction)")
            
        except Exception as e:
            logger.error(f"Failed to convert {filename}: {e}")

    logger.info(f"Migration complete in {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    convert_to_parquet()

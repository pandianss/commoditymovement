import os
import pickle
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("cache_manager", log_file="cache.log")

class CacheManager:
    """
    Manages caching of processed data to avoid redundant computation.
    Uses content hashing to detect changes and invalidate cache.
    """
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.pkl"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _compute_hash(self, filepath):
        """Compute MD5 hash of a file."""
        if not os.path.exists(filepath):
            return None
        
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_cache_valid(self, cache_key, source_files):
        """
        Check if cache is valid by comparing source file hashes.
        
        Args:
            cache_key: Unique identifier for the cached item
            source_files: List of source file paths that the cache depends on
        
        Returns:
            bool: True if cache is valid, False otherwise
        """
        if cache_key not in self.metadata:
            logger.debug(f"Cache miss: {cache_key} not in metadata")
            return False
        
        cached_meta = self.metadata[cache_key]
        
        # Check if all source files still exist and match hashes
        for filepath in source_files:
            current_hash = self._compute_hash(filepath)
            if current_hash != cached_meta.get('source_hashes', {}).get(filepath):
                logger.debug(f"Cache invalid: {filepath} has changed")
                return False
        
        # Check if cache file exists
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            logger.debug(f"Cache miss: {cache_file} does not exist")
            return False
        
        logger.info(f"Cache hit: {cache_key}")
        return True
    
    def get(self, cache_key):
        """Retrieve cached data."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded from cache: {cache_key}")
            return data
        return None
    
    def set(self, cache_key, data, source_files):
        """
        Store data in cache with metadata.
        
        Args:
            cache_key: Unique identifier for the cached item
            data: Data to cache (must be picklable)
            source_files: List of source file paths that the cache depends on
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Save data
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Update metadata
        source_hashes = {fp: self._compute_hash(fp) for fp in source_files}
        self.metadata[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'source_hashes': source_hashes,
            'cache_file': str(cache_file)
        }
        self._save_metadata()
        
        logger.info(f"Cached: {cache_key}")
    
    def invalidate(self, cache_key):
        """Remove a specific cache entry."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()
        logger.info(f"Invalidated cache: {cache_key}")
    
    def clear_all(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.name != "cache_metadata.pkl":
                cache_file.unlink()
        self.metadata = {}
        self._save_metadata()
        logger.info("Cleared all cache")
    
    def get_stats(self):
        """Get cache statistics."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        return {
            'num_entries': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

# Example usage
if __name__ == "__main__":
    cache = CacheManager()
    
    # Example: Cache processed features
    source_files = ["data/raw/commodities_raw.csv", "data/raw/macro_raw.csv"]
    cache_key = "feature_store_v1"
    
    if cache.is_cache_valid(cache_key, source_files):
        data = cache.get(cache_key)
        print(f"Loaded {len(data)} rows from cache")
    else:
        # Simulate processing
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        cache.set(cache_key, data, source_files)
        print("Processed and cached data")
    
    print(cache.get_stats())

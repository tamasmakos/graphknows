import os
import csv
import logging
from datetime import datetime
from typing import List

from .synthetic_gen import run_synthetic_generation

logger = logging.getLogger(__name__)

def generate_and_split_data(days: int, start_date_str: str, buffer_dir: str = "buffer") -> List[str]:
    """
    Generates synthetic life-log data for N days and splits it into daily files.
    
    Args:
        days: Number of days to simulate.
        start_date_str: Start date (YYYY-MM-DD).
        buffer_dir: Directory to save intermediate files.
        
    Returns:
        List of absolute paths to daily CSV files, ordered by date.
    """
    # ensure buffer directory exists
    os.makedirs(buffer_dir, exist_ok=True)
    
    full_log_path = os.path.join(buffer_dir, "full_log.csv")
    
    # 1. Run Generation
    # Note: run_synthetic_generation handles clearing existing file if present
    logger.info(f"🧬 Generating {days} days of synthetic data starting {start_date_str}...")
    run_synthetic_generation(output_csv=full_log_path, start_date_str=start_date_str, num_days=days)
    
    # 2. Split by Day
    logger.info("🔪 Splitting data into daily chunks...")
    daily_files = {} # date_str -> valid rows
    
    # Read full log
    if not os.path.exists(full_log_path):
        raise RuntimeError(f"Generation failed: {full_log_path} not found")

    with open(full_log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            # Parse time to get date
            # Time format from generator: '%Y-%m-%d %H:%M:%S'
            try:
                time_str = row['Time']
                dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                date_key = dt.strftime('%Y-%m-%d')
                
                if date_key not in daily_files:
                    daily_files[date_key] = []
                daily_files[date_key].append(row)
            except ValueError:
                logger.warning(f"Skipping row with invalid time format: {row.get('Time')}")
                continue

    # Write daily files
    sorted_dates = sorted(daily_files.keys())
    created_files = []
    
    for i, date_key in enumerate(sorted_dates):
        # Naming convention: day_{i+1:03d}_{date}.csv
        filename = f"day_{i+1:03d}_{date_key}.csv"
        filepath = os.path.join(buffer_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(daily_files[date_key])
            
        logger.info(f"  Saved Day {i+1} ({date_key}): {len(daily_files[date_key])} rows -> {filepath}")
        created_files.append(os.path.abspath(filepath))
        
    return created_files

def generate_single_day(date: datetime, output_dir: str = "buffer") -> str:
    """
    Generate synthetic data for a single day.
    
    Args:
        date: The date to simulate.
        output_dir: Directory to save the file.
        
    Returns:
        Absolute path to the generated CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = date.strftime('%Y-%m-%d')
    
    # Naming convention matches orchestrator expectations
    # day_{i+1:03d}_{date}.csv ... actually orchestrator handles day index.
    # Adapter just returning a file path is better.
    # But wait, orchestrator loop determines 'i'. 
    # Let's just name it by date: `day_log_{date}.csv`
    
    filename = f"day_log_{date_str}.csv"
    filepath = os.path.join(output_dir, filename)
    
    logger.info(f"🧬 Generating single day log for {date_str} -> {filepath}")
    
    # Run generation for 1 day
    # synthetic_gen's run_synthetic_generation clears the file if it exists.
    run_synthetic_generation(output_csv=filepath, start_date_str=date_str, num_days=1)
    
    return os.path.abspath(filepath)

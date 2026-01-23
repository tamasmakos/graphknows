"""Life Log CSV Parser.

Parses CSV logs containing Time, Location, Audio, and Image columns.
Groups logs into time-based segments (episodes) based on a 5-minute gap threshold.
"""

import csv
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from src.kg.graph.parsers.base import BaseDocumentParser
from src.kg.graph.parsing import SegmentData

logger = logging.getLogger(__name__)

class LifeLogParser(BaseDocumentParser):
    """Parser for Life Log CSV format.
    
    Expected Columns: Time, Location, Audio, Image
    """
    
    def parse(self, content: str, filename: str, doc_date: date) -> List[SegmentData]:
        """Parse CSV content into segments (episodes)."""
        segments = []
        rows = []
        
        # Parse CSV content
        try:
            # splitting by lines first to handle potential issues, but csv.DictReader expects iterable
            lines = content.strip().split('\n')
            reader = csv.DictReader(lines)
            
            for row in reader:
                # Basic validation
                if not all(k in row for k in ['Time', 'Location', 'Audio', 'Image']):
                    continue
                rows.append(row)
        except Exception as e:
            logger.error(f"Failed to parse CSV content: {e}")
            return []
            
        if not rows:
            return []
            
        # Group into segments based on 5-minute gap
        current_segment_rows = []
        last_time = None
        
        # Sort rows by time just in case
        try:
            rows.sort(key=lambda x: self._parse_time(x['Time']))
        except ValueError:
             # Fallback if time format is inconsistent, trust order
             pass
             
        segment_idx = 0
        
        for row in rows:
            try:
                row_time = self._parse_time(row['Time'])
            except ValueError:
                logger.warning(f"Skipping row with invalid time: {row.get('Time')}")
                continue
                
            if last_time is None:
                current_segment_rows.append(row)
                last_time = row_time
                continue
                
            # Check gap
            gap = row_time - last_time
            if gap > timedelta(minutes=5):
                # Finalize current segment
                if current_segment_rows:
                    segments.append(self._create_segment(current_segment_rows, doc_date, segment_idx))
                    segment_idx += 1
                current_segment_rows = [row]
            else:
                current_segment_rows.append(row)
            
            last_time = row_time
            
        # Add last segment
        if current_segment_rows:
             segments.append(self._create_segment(current_segment_rows, doc_date, segment_idx))
             
        return segments
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string with multiple format fallbacks."""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%A %d %B %Y, %H:%M",  # Tuesday 23 December 2025, 07:46
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Could not parse time: {time_str}")

    def _create_segment(self, rows: List[Dict], doc_date: date, idx: int) -> SegmentData:
        """Helper to create a SegmentData object from a list of rows."""
        start_time_str = rows[0]['Time']
        end_time_str = rows[-1]['Time']
        
        # Generate ID
        # SEGMENT_{DATE}_EPISODE_{ID}
        segment_id = f"SEGMENT_{doc_date.isoformat()}_EPISODE_{idx:03d}"
        
        # Summary content (could be improved explicitly later)
        # Concatenate audio for simple text representation
        # Truncate to avoid massive nodes if many rows
        content_summary = "\n".join([f"[{r['Time'].split(' ')[1]}] {r['Location']}: {r['Audio'][:50]}..." for r in rows])
        
        # Identify locations in this segment
        locations = list(set([r['Location'] for r in rows if r['Location']]))
        
        return SegmentData(
            segment_id=segment_id,
            content=content_summary,
            date=doc_date,
            line_number=idx, # Using idx as logical line number
            metadata={
                'type': 'EPISODE',
                'start_time': start_time_str,
                'end_time': end_time_str,
                'locations': locations,
                'conversations': rows  # Store full rows for detailed extraction
            }
        )

    def extract_date(self, filename: str) -> Optional[str]:
        """Extract date from filename (e.g., life_log_2024-01-01.csv)."""
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            return match.group(1)
        return None
    
    def supports_file(self, filename: str) -> bool:
        return filename.lower().endswith('.csv')

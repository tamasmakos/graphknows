import os
import random
import logging
from typing import List, Dict, Any
from datetime import datetime, date
from graphgen.data_types import SegmentData
from graphgen.config.settings import IterativeSettings

logger = logging.getLogger(__name__)

class IterativeLoader:
    """
    Loads and samples speeches (lines) from text files for iterative processing.
    """
    def __init__(self, input_dir: str, settings: IterativeSettings, file_pattern: str = "*.txt"):
        self.input_dir = input_dir
        self.settings = settings
        self.file_pattern = file_pattern
        self.all_speeches: List[Dict[str, Any]] = []
        self._load_all_speeches()
        
    def _load_all_speeches(self):
        """Read all lines from all text files in input_dir."""
        import fnmatch
        logger.info(f"Loading all speeches from {self.input_dir} matching '{self.file_pattern}'...")
        if not os.path.exists(self.input_dir):
            logger.error(f"Input directory {self.input_dir} does not exist.")
            return

        for filename in os.listdir(self.input_dir):
            if not fnmatch.fnmatch(filename, self.file_pattern):
                continue
                
            file_path = os.path.join(self.input_dir, filename)
            try:
                # No date parsing as requested
                doc_date = None

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        
                        self.all_speeches.append({
                            'content': line,
                            'filename': filename,
                            'line_number': line_idx,
                            'date': doc_date
                        })
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
        
        logger.info(f"Loaded {len(self.all_speeches)} total speeches.")

    def get_batch(self, iteration: int) -> List[SegmentData]:
        """
        Get a random batch of speeches.
        Uses seed + iteration to ensure reproducibility per iteration but randomness between them.
        """
        if not self.all_speeches:
            return []
            
        # Set seed for this iteration
        random.seed(self.settings.random_seed + iteration)
        
        # Sample with replacement? Or without? 
        # "100 speeches r n times". Typically implies sampling from the population.
        # Let's do random sampling without replacement for the batch, 
        # but the pool remains constant (so batches can overlap if n is large, or be distinct).
        # Standard bootstrapping is with replacement. 
        # But for coverage, usually we want distinct sets if possible, or just random samples.
        # I'll use random.sample (without replacement within batch).
        
        batch_size = min(self.settings.batch_size, len(self.all_speeches))
        sampled_dicts = random.sample(self.all_speeches, batch_size)
        
        segments = []
        for i, s in enumerate(sampled_dicts):
            # Create a unique segment ID for this iteration to avoid graph collisions if reusing graph?
            # Or should we clear graph between iterations?
            # The prompt says "iteratively upload...". If "upload" means "add to graph", 
            # maybe it means Cumulative?
            # "iteratively upload 100 speeches... One line is one speech".
            # "help me prove that with better modularity scores... the further the topic summary embeddings are."
            # This implies strictly separate experiments or a cumulative graph?
            # If cumulative, modularity usually drops or stabilizes.
            # If independent, we compare different graph structures.
            # "snap a picture... for each community".
            # I will assume Independent runs (clearing graph) to get clean stats for that batch size/configuration,
            # OR typically "iteratively" might mean "Add 100, measure. Add another 100, measure."
            # "iteratively upload... of the total speeches".
            # Let's implement Cumulative addition. It's more interesting for "iteratively".
            # BUT, for "proving correlation", independent samples are statistically cleaner.
            # However, "Iteratively upload" strongly suggests growing the graph.
            # I will generate unique IDs so they can coexist if needed, but I'll likely clear or accumulate based on orchestrator logic.
            # Let's stick to unique IDs.
            
            segment_id = f"SEG_{iteration}_{i}_{s['filename']}_{s['line_number']}"
            
            segments.append(SegmentData(
                segment_id=segment_id,
                content=s['content'],
                line_number=s['line_number'],
                date=s['date'],
                metadata={'filename': s['filename'], 'iteration': iteration}
            ))
            
        return segments

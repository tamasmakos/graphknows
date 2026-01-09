import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kg.config import KGConfig, DEFAULT_CONFIG
from src.kg.pipeline import build_semantic_kg_with_communities

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    # Setup test config
    config_dict = DEFAULT_CONFIG.copy()
    config_dict['input_dir'] = '/workspaces/kg/input/test'
    config_dict['output_dir'] = '/workspaces/kg/output/test_verification'
    config_dict['speech_limit'] = 1 # Process only 1 speech for quick testing (saves API tokens)
    config_dict['llm_model'] = 'groq/llama-3.3-70b-versatile' # Use Groq model
    
    config = KGConfig(**config_dict)
    
    print(f"Running pipeline with input: {config.get('input_dir')}")
    print(f"Output directory: {config.get('output_dir')}")
    
    try:
        results = await build_semantic_kg_with_communities(
            input_dir=config.get('input_dir'),
            output_dir=config.get('output_dir'),
            config=config
        )
        
        print("\nPipeline completed successfully!")
        print(f"Lexical Stats: {results.lexical_graph_stats}")
        print(f"Extraction Stats: {results.extraction_stats}")
        print(f"Community Stats: {results.community_stats}")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

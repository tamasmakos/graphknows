
import asyncio
import logging
from kg.graph.extractors import LangChainExtractor
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_extraction():
    config = {
        'extraction': {
            'backend': 'langchain'
        },
        'llm': {
            'type': 'openai', # Will use environment variables or mock if not available
            'model': 'gpt-4o'
        }
    }
    
    # Mocking get_langchain_llm if necessary, but assuming environment is set up as user has running services
    # If this fails due to missing keys, we might need to rely on existing running services log
    
    try:
        extractor = LangChainExtractor(config)
        text = "John Doe works at Google as a Software Engineer. He lives in San Francisco."
        
        print("Starting extraction...")
        relations, nodes = await extractor.extract_relations(text)
        
        print("\n--- Relations ---")
        for r in relations:
            print(r)
            
        print("\n--- Nodes ---")
        for n in nodes:
            print(n)
            
        # Verify structure
        assert isinstance(relations, list)
        assert isinstance(nodes, list)
        if nodes:
            assert 'id' in nodes[0]
            assert 'type' in nodes[0]
            print("\n✅ Verification Successful: Nodes extracted with types!")
        else:
            print("\n⚠️ Verification Warning: No nodes extracted (might be LLM fluke or config)")

    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_extraction())

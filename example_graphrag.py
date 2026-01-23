import sys
import os
import asyncio
from dotenv import load_dotenv

# Add services/graphrag to path so 'src' package is found
sys.path.insert(0, os.path.abspath("services/graphrag"))

# Load environment variables from .env file
load_dotenv()

# Default to localhost for external script execution if not set
os.environ.setdefault("FALKORDB_HOST", "localhost")
os.environ.setdefault("FALKORDB_PORT", "6379")

try:
    from src.workflow.graph_workflow import GraphWorkflow
    print("[GraphRAG] Imports successful.")
except ImportError as e:
    print(f"[GraphRAG] Import failed: {e}")
    sys.exit(1)

async def run_example():
    print("[GraphRAG] Initializing Workflow...")
    
    # Instantiate the workflow
    # timeout: seconds to wait for the workflow to complete
    # verbose: print step execution details
    workflow = GraphWorkflow(timeout=60, verbose=True)
    
    # Get query from command line args or use default
    query = "How could I improve my morning routine?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
    print(f"[GraphRAG] Running query: {query}")
    print("-" * 50)
    
    try:
        # Run the workflow
        result = await workflow.run(query=query)
        
        # Display Results
        print("\n" + "="*50)
        print(result.get("answer"))
        
        print("\n" + "="*50)
        print("🧠 REASONING TRACE")
        print("="*50)
        for step_info in result.get("trace", []):
            print(f"• {step_info}")

        # Optional: Print Context if needed
        # print("\n" + "="*50)
        # print("📄 CONTEXT USED")
        # print("="*50)
        # print(result.get("context", "")[:1000] + "...")
        
    except Exception as e:
        print(f"\n[GraphRAG] ❌ Execution failed: {e}")
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_example())

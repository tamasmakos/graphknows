import time
import asyncio
from gliner import GLiNER

def benchmark():
    print("Loading GLiNER model...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    labels = ["Person", "Organization", "Location", "Event", "Date", "Award", "Competitions", "Teams", "Concept"]
    
    # Create sample data (100 sentences)
    sentence = "Apple Inc. was founded by Steve Jobs in Cupertino."
    sentences = [sentence] * 10
    print(f"Benchmarking with {len(sentences)} sentences...")

    # 1. Sequential (Current Approach Simulation - Single Threaded for simplicity, as thread overhead is minimal compared to model inference)
    # Note: The real app uses run_in_executor, which might add overhead or help slightly if GIL is released, but batching is key.
    start_time = time.time()
    for s in sentences:
        model.predict_entities(s, labels, threshold=0.5)
    seq_time = time.time() - start_time
    print(f"Sequential/Individual time: {seq_time:.4f}s")

    # 2. Batched
    start_time = time.time()
    if hasattr(model, 'batch_predict_entities'):
        print("Using batch_predict_entities")
        model.batch_predict_entities(sentences, labels, threshold=0.5)
    elif hasattr(model, 'predict_entities_batch'):
        print("Using predict_entities_batch")
        model.predict_entities_batch(sentences, labels, threshold=0.5)
    else:
        # Try passing list to predict_entities
        try:
            print("Trying predict_entities with list")
            model.predict_entities(sentences, labels, threshold=0.5)
        except Exception as e:
            print(f"Batching not supported directly: {e}")
            return

    batch_time = time.time() - start_time
    print(f"Batched time: {batch_time:.4f}s")
    
    print(f"Speedup: {seq_time / batch_time:.2f}x")

if __name__ == "__main__":
    benchmark()

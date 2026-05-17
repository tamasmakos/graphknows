"""Entrypoint for HybridRAG service."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "hybridrag.api:app",
        host="0.0.0.0",
        port=8020,
        reload=False,
        log_level="info",
    )

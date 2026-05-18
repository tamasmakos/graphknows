"""graphknows service entrypoint — uvicorn on port 8025."""
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "graphknows.api:app",
        host="0.0.0.0",
        port=8025,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()

"""``python -m graphknows.integrations.claude_code <verb>`` — the hook entry point.

The shipped ``settings.json`` names the package, not the module, so the command a
user copies keeps working if the verb implementation ever moves.
"""

from __future__ import annotations

import sys

from graphknows.integrations.claude_code.hooks import main

if __name__ == "__main__":
    sys.exit(main())

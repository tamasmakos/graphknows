"""
Deprecated shim — SessionGraphManager moved to agent.session.manager.

This module is kept for backward compatibility only.
New code should import from agent.session.manager directly.
"""
from agent.session.manager import SessionGraphManager  # noqa: F401

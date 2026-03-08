"""MCP tool registration for the CLI server."""

from .claude import register_claude_tools
from .gemini import register_gemini_tools
from .orchestrate import register_orchestrate
from .session_usage import register_session_usage_tools


def register_all(mcp):
    """Register all tools on the given FastMCP instance."""
    register_gemini_tools(mcp)
    register_claude_tools(mcp)
    register_orchestrate(mcp)
    register_session_usage_tools(mcp)

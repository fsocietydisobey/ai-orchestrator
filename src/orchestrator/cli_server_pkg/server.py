"""FastMCP server and entry point for the CLI-based MCP server."""

from mcp.server.fastmcp import FastMCP

from .tools import register_all

mcp = FastMCP("ai-orchestrator")
register_all(mcp)


def main() -> None:
    """Entry point — run the MCP server over stdio."""
    mcp.run()

"""Configuration for the CLI-based MCP server (env-based)."""

import os

# Project root defaults to cwd; override with PROJECT_ROOT env var
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())

# Subprocess timeout in seconds (5 minutes default)
CLI_TIMEOUT = int(os.environ.get("CLI_TIMEOUT", "300"))

# Models — override via env vars
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

# CLI paths — override via env vars if needed
CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

# Gemini runs via npx under nvm — we need the absolute path to npx
# since nvm isn't loaded in subprocess shells
_NVM_NODE_VERSION = os.environ.get("NVM_NODE_VERSION", "24.12.0")
_NVM_BIN = os.path.expanduser(f"~/.nvm/versions/node/v{_NVM_NODE_VERSION}/bin")
NPX_CMD = os.environ.get("NPX_CMD", os.path.join(_NVM_BIN, "npx"))
GEMINI_PKG = os.environ.get("GEMINI_PKG", "@google/gemini-cli@latest")

# Total timeout for the entire orchestration pipeline (default 10 minutes)
ORCHESTRATE_TIMEOUT = int(os.environ.get("ORCHESTRATE_TIMEOUT", "600"))

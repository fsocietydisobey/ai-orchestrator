"""CLI subprocess execution helpers."""

import asyncio
import os
import shutil

from .. import config

# Track the currently running subprocess so it can be killed on cancellation
active_process: asyncio.subprocess.Process | None = None


def cli_available(cmd: str) -> bool:
    """Check if a CLI tool exists (absolute path or on PATH)."""
    if os.path.isabs(cmd):
        return os.path.isfile(cmd) and os.access(cmd, os.X_OK)
    return shutil.which(cmd) is not None


def kill_active_process():
    """Kill the currently running subprocess if any."""
    global active_process
    if active_process and active_process.returncode is None:
        active_process.kill()
        active_process = None


async def run_cli(cmd: list[str], timeout: int | None = None) -> str:
    """Run a CLI command as a subprocess and return stdout.

    Args:
        cmd: Command and arguments to run.
        timeout: Max seconds to wait before killing the process (default: config.CLI_TIMEOUT).

    Returns:
        The process stdout as a string.

    Raises:
        TimeoutError: If the process exceeds the timeout.
        RuntimeError: If the process exits with a non-zero code.
    """
    global active_process

    if timeout is None:
        timeout = config.CLI_TIMEOUT

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=config.PROJECT_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    active_process = proc

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise TimeoutError(
            f"CLI command timed out after {timeout}s: {' '.join(cmd[:2])}..."
        )
    finally:
        active_process = None

    if proc.returncode != 0:
        err = stderr.decode().strip()
        raise RuntimeError(
            f"CLI exited with code {proc.returncode}: {err or '(no stderr)'}"
        )

    return stdout.decode()

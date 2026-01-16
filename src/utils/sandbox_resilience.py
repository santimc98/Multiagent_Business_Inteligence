"""
Sandbox resilience utilities.

Provides retry logic and robustness for transient failures in sandbox operations.
"""

import re
import time
import inspect
from typing import Optional, Any, Callable


# Patterns that suggest transient/temporary errors
TRANSIENT_ERROR_PATTERNS = [
    "timeout", "timed out", "temporarily unavailable",
    "connection", "expired", "gone",
    "502", "503", "504",
    "connection reset", "broken pipe",
]


def _is_transient_error(error_message: str) -> bool:
    """
    Check if an error message suggests a transient/temporary failure.

    Args:
        error_message: The error message to check

    Returns:
        True if the error appears transient
    """
    error_lower = error_message.lower()
    return any(pattern in error_lower for pattern in TRANSIENT_ERROR_PATTERNS)


def run_code_with_optional_timeout(sandbox: Any, code: str, timeout_s: Optional[int] = None) -> Any:
    """
    Run code in sandbox with optional timeout.

    Uses inspect to check if sandbox.run_code accepts timeout parameter.

    Args:
        sandbox: The sandbox instance
        code: The code to run
        timeout_s: Optional timeout in seconds

    Returns:
        Result from sandbox.run_code
    """
    sig = inspect.signature(sandbox.run_code)

    if "timeout" in sig.parameters and timeout_s is not None:
        return sandbox.run_code(code, timeout=timeout_s)
    else:
        return sandbox.run_code(code)


def run_cmd_with_retry(sandbox: Any, cmd: str, retries: int = 2) -> Any:
    """
    Run a sandbox command with retry logic for transient errors.

    Retries if the error appears to be transient.

    Args:
        sandbox: The sandbox instance
        cmd: The command to run
        retries: Number of retries (default: 2)

    Returns:
        Result from sandbox.commands.run

    Raises:
        Last exception if all retries fail
    """
    last_error = None

    for attempt in range(retries + 1):
        try:
            return sandbox.commands.run(cmd)
        except Exception as e:
            last_error = e
            error_msg = str(e)

            if attempt < retries and _is_transient_error(error_msg):
                # Exponential backoff
                backoff = 2 ** attempt
                print(f"RETRYING_SANDBOX_CMD (attempt {attempt + 1}/{retries}): {error_msg}")
                time.sleep(backoff)
            else:
                # Not transient or out of retries
                raise

    # All retries exhausted
    if last_error is not None:
        raise last_error
    raise Exception("All retries exhausted without specific error")


def safe_download_file(sandbox: Any, remote_path: str, local_path: str) -> Optional[str]:
    """
    Download a file from sandbox with base64 fallback.

    Args:
        sandbox: The sandbox instance
        remote_path: Path in sandbox to download
        local_path: Local path to save to

    Returns:
        Content as string, or None if download fails
    """
    try:
        return sandbox.files.read(remote_path)
    except Exception as e:
        print(f"SANDBOX_DOWNLOAD_FAILED: {e}")

        # Try base64 fallback
        try:
            cmd = f"base64 -w 0 {remote_path}"
            result = sandbox.commands.run(cmd)
            if result and result.strip():
                import base64
                decoded = base64.b64decode(result.strip())
                return decoded.decode("utf-8", errors="ignore")
        except Exception as e2:
            print(f"SANDBOX_BASE64_FALLBACK_FAILED: {e2}")

        return None

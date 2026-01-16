"""
Tests for sandbox_resilience module.

Tests retry logic, timeout support, and download fallbacks.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.utils.sandbox_resilience import (
    is_transient_error_like,
    is_transient_sandbox_error,
    run_code_with_optional_timeout,
    run_cmd_with_retry,
    safe_download_bytes,
)


def test_is_transient_error_like_timeout():
    """Test that timeout errors are detected as transient."""
    err_msg = "Script execution timed out after 10.0 seconds"

    assert is_transient_error_like(err_msg) is True


def test_is_transient_error_like_503():
    """Test that 503 errors are detected as transient."""
    err_msg = "503 Service Unavailable"

    assert is_transient_error_like(err_msg) is True


def test_is_transient_error_like_connection():
    """Test that connection errors are detected as transient."""
    err_msg = "Connection reset by peer"

    assert is_transient_error_like(err_msg) is True


def test_is_transient_error_like_sandbox_expired():
    """Test that sandbox expired errors are detected as transient."""
    err_msg = "sandbox expired: container was terminated"

    assert is_transient_sandbox_error(err_msg) is True


def test_is_transient_sandbox_error_non_transient():
    """Test that script syntax errors are NOT transient."""
    err_msg = "SyntaxError: invalid syntax"

    assert is_transient_sandbox_error(err_msg) is False
    assert is_transient_error_like(err_msg) is False


def test_run_code_with_optional_timeout():
    """Test run_code with timeout when sandbox supports it."""
    mock_sandbox = Mock()

    # Sandbox that supports timeout
    sig_mock = Mock()
    sig_mock.parameters = {"timeout": {"name": "timeout"}}
    mock_sandbox.run_code = Mock(return_value=MagicMock())
    mock_sandbox.run_code.__signature__ = sig_mock

    result = run_code_with_optional_timeout(mock_sandbox, "print('hello')", timeout_s=60)
    assert mock_sandbox.run_code.called
    mock_sandbox.run_code.assert_called_with("print('hello')", timeout=60)


def test_run_code_with_optional_timeout_no_timeout_param():
    """Test run_code without timeout when sandbox doesn't support it."""
    mock_sandbox = Mock()
    sig_mock = Mock()
    sig_mock.parameters = {}  # No timeout parameter
    mock_sandbox.run_code = Mock(return_value=MagicMock())
    mock_sandbox.run_code.__signature__ = sig_mock

    result = run_code_with_optional_timeout(mock_sandbox, "print('hello')", timeout_s=None)
    assert mock_sandbox.run_code.called
    mock_sandbox.run_code.assert_called_with("print('hello')")


def test_run_cmd_with_retry_transient_success():
    """Test that run_cmd_with_retry succeeds on second attempt after transient error."""
    mock_sandbox = Mock()
    cmd = "ls -la"

    # First attempt fails with transient error
    mock_sandbox.commands.run = Mock(side_effect=[Exception("503 Service Unavailable")])

    # Second attempt succeeds
    def side_effect_func(attempt_num):
        if attempt_num == 1:
            raise Exception("503 Service Unavailable")
        else:
            result = Mock()
            result.stdout = "file1.txt\nfile2.txt\n"
            result.exit_code = 0
            return result

    mock_sandbox.commands.run = Mock(side_effect=side_effect_func)

    # Patch time.sleep to speed up test
    import time
    original_sleep = time.sleep
    time.sleep = Mock()

    run_cmd_with_retry(mock_sandbox, cmd, retries=2)
    time.sleep = original_sleep

    # Should have called twice and succeeded
    assert mock_sandbox.commands.run.call_count == 2


def test_run_cmd_with_retry_non_transient_failure():
    """Test that run_cmd_with_retry raises immediately on non-transient error."""
    mock_sandbox = Mock()
    cmd = "ls -la"

    # Always fails with syntax error
    mock_sandbox.commands.run = Mock(side_effect=Exception("SyntaxError: invalid syntax"))

    run_cmd_with_retry(mock_sandbox, cmd, retries=2)

    # Should have called only once
    assert mock_sandbox.commands.run.call_count == 1


def test_safe_download_bytes_success():
    """Test successful file download with bytes return."""
    mock_sandbox = Mock()
    remote_path = "/path/to/file.txt"

    mock_sandbox.files.read = Mock(return_value=b"file content")
    result = safe_download_bytes(mock_sandbox, remote_path, max_attempts=2)

    assert result == b"file content"


def test_safe_download_bytes_with_base64_fallback():
    """Test download with base64 fallback when files.read fails."""
    mock_sandbox = Mock()
    remote_path = "/path/to/file.txt"

    # files.read fails on first attempt
    # base64 fallback succeeds on second
    cmd_result = Mock()
    cmd_result.stdout = "ZmlsZGllpcg=="

    def side_effect_func(attempt_num):
        if attempt_num == 1:
            raise Exception("files.read failed")
        else:
            mock_sandbox.commands.run = Mock(return_value=cmd_result)
            return cmd_result

    mock_sandbox.files.read = Mock(side_effect=side_effect_func)
    mock_sandbox.commands.run = Mock(return_value=cmd_result)

    result = safe_download_bytes(mock_sandbox, remote_path, max_attempts=2)

    assert result == "file content"  # Decoded base64 result

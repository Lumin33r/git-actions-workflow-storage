"""
Performance tests for Ollama service response times and baselines.

This module contains tests that measure and validate the performance
characteristics of the Ollama service, including response times,
cold start performance, and cache effectiveness.
"""

import getpass
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest


# Performance thresholds (in seconds)
MAX_RESPONSE_TIME = 30      # Hard fail if query exceeds this
WARN_RESPONSE_TIME = 15     # Advisory warning if query exceeds this
MAX_COLD_START_TIME = 45    # Maximum allowed time for cold start

# Log file path
PERFORMANCE_LOG_FILE = Path(__file__).parent / "performance_log.txt"


def log_timing(test_name: str, message: str) -> None:
    """
    Log timing information to the performance log file.

    Each entry includes timestamp, username, test name, and message.

    Args:
        test_name: Name of the test function.
        message: The timing message to log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = getpass.getuser()

    log_entry = f"[{timestamp}] User: {user} | Test: {test_name} | {message}\n"

    # Append to log file
    with open(PERFORMANCE_LOG_FILE, "a") as f:
        f.write(log_entry)

    # Also print to console
    print(f"\n{message}")


@pytest.mark.critical
def test_ai_response_time(model_name, sample_prompt):
    """
    Verify that AI query completes within 30 seconds.

    This test measures the time taken for a standard query
    and fails if it exceeds the maximum threshold.

    Args:
        model_name: Fixture providing the model name.
        sample_prompt: Fixture providing a test prompt.
    """
    start_time = time.time()

    result = subprocess.run(
        ['ollama', 'run', model_name, sample_prompt],
        capture_output=True,
        text=True,
        timeout=60
    )

    elapsed_time = time.time() - start_time

    assert result.returncode == 0, (
        f"Query failed with exit code {result.returncode}. "
        f"stderr: {result.stderr}"
    )

    log_timing("test_ai_response_time", f"AI response time: {elapsed_time:.2f}s")

    assert elapsed_time < MAX_RESPONSE_TIME, (
        f"AI response took {elapsed_time:.2f}s, "
        f"exceeding maximum threshold of {MAX_RESPONSE_TIME}s"
    )


@pytest.mark.advisory
def test_ai_response_time_warning(model_name, sample_prompt):
    """
    Warn if AI query takes longer than 15 seconds.

    This advisory test provides a warning when response time
    exceeds the optimal threshold, but does not fail the test.

    Args:
        model_name: Fixture providing the model name.
        sample_prompt: Fixture providing a test prompt.
    """
    start_time = time.time()

    result = subprocess.run(
        ['ollama', 'run', model_name, sample_prompt],
        capture_output=True,
        text=True,
        timeout=60
    )

    elapsed_time = time.time() - start_time

    assert result.returncode == 0, (
        f"Query failed with exit code {result.returncode}. "
        f"stderr: {result.stderr}"
    )

    log_timing("test_ai_response_time_warning", f"AI response time: {elapsed_time:.2f}s")

    if elapsed_time > WARN_RESPONSE_TIME:
        pytest.skip(
            f"ADVISORY: Response time {elapsed_time:.2f}s exceeds "
            f"optimal threshold of {WARN_RESPONSE_TIME}s"
        )


@pytest.mark.critical
def test_model_load_time(model_name, sample_prompt):
    """
    Verify that first query (cold start) completes within 45 seconds.

    The first query may include model loading time, so we allow
    a longer threshold for cold start scenarios.

    Args:
        model_name: Fixture providing the model name.
        sample_prompt: Fixture providing a test prompt.
    """
    start_time = time.time()

    result = subprocess.run(
        ['ollama', 'run', model_name, sample_prompt],
        capture_output=True,
        text=True,
        timeout=90
    )

    elapsed_time = time.time() - start_time

    assert result.returncode == 0, (
        f"Cold start query failed with exit code {result.returncode}. "
        f"stderr: {result.stderr}"
    )

    log_timing("test_model_load_time", f"Model load time (cold start): {elapsed_time:.2f}s")

    assert elapsed_time < MAX_COLD_START_TIME, (
        f"Cold start took {elapsed_time:.2f}s, "
        f"exceeding maximum threshold of {MAX_COLD_START_TIME}s"
    )


@pytest.mark.advisory
def test_cache_improves_performance(model_name, sample_prompt, test_output_dir):
    """
    Verify that second query is faster than first query.

    After the model is loaded into memory, subsequent queries
    should complete faster due to caching effects.

    Args:
        model_name: Fixture providing the model name.
        sample_prompt: Fixture providing a test prompt.
        test_output_dir: Fixture providing output directory for timing report.
    """
    # First query (cold or warm start)
    start_time_1 = time.time()

    result_1 = subprocess.run(
        ['ollama', 'run', model_name, sample_prompt],
        capture_output=True,
        text=True,
        timeout=90
    )

    elapsed_time_1 = time.time() - start_time_1

    assert result_1.returncode == 0, (
        f"First query failed with exit code {result_1.returncode}. "
        f"stderr: {result_1.stderr}"
    )

    # Second query (should be cached/warm)
    start_time_2 = time.time()

    result_2 = subprocess.run(
        ['ollama', 'run', model_name, sample_prompt],
        capture_output=True,
        text=True,
        timeout=60
    )

    elapsed_time_2 = time.time() - start_time_2

    assert result_2.returncode == 0, (
        f"Second query failed with exit code {result_2.returncode}. "
        f"stderr: {result_2.stderr}"
    )

    # Save timing report
    timing_report = test_output_dir / "timing_report.txt"
    timing_report.write_text(
        f"Cache Performance Report\n"
        f"========================\n"
        f"Model: {model_name}\n"
        f"First query time: {elapsed_time_1:.3f}s\n"
        f"Second query time: {elapsed_time_2:.3f}s\n"
        f"Improvement: {elapsed_time_1 - elapsed_time_2:.3f}s\n"
        f"Cache effective: {elapsed_time_2 < elapsed_time_1}\n"
    )

    log_timing("test_cache_improves_performance", f"First query time: {elapsed_time_1:.2f}s")
    log_timing("test_cache_improves_performance", f"Second query time: {elapsed_time_2:.2f}s")
    log_timing("test_cache_improves_performance", f"Timing report saved to: {timing_report}")

    assert elapsed_time_2 < elapsed_time_1, (
        f"Cache did not improve performance. "
        f"First query: {elapsed_time_1:.2f}s, "
        f"Second query: {elapsed_time_2:.2f}s"
    )


@pytest.mark.critical
def test_response_not_empty(model_name, sample_prompt):
    """
    Verify that AI response contains actual content.

    This test ensures the model produces meaningful output,
    not just empty or whitespace-only responses.

    Args:
        model_name: Fixture providing the model name.
        sample_prompt: Fixture providing a test prompt.
    """
    result = subprocess.run(
        ['ollama', 'run', model_name, sample_prompt],
        capture_output=True,
        text=True,
        timeout=60
    )

    assert result.returncode == 0, (
        f"Query failed with exit code {result.returncode}. "
        f"stderr: {result.stderr}"
    )

    response = result.stdout.strip()

    log_timing("test_response_not_empty", f"Response length: {len(response)} characters")

    assert len(response) > 0, (
        f"AI response is empty. "
        f"Expected actual content for prompt: '{sample_prompt}'"
    )

    assert not response.isspace(), (
        f"AI response contains only whitespace. "
        f"Expected meaningful content."
    )

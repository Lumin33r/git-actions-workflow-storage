"""
Shared test fixtures and configuration for Ollama service tests.

This module provides reusable pytest fixtures for testing Ollama functionality,
including service availability checks, model configuration, and output directories.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Generator

import pytest


def pytest_configure(config):
    """Register custom pytest markers for test categorization."""
    config.addinivalue_line(
        "markers",
        "critical: marks tests as critical - must pass for service to be considered healthy"
    )
    config.addinivalue_line(
        "markers",
        "advisory: marks tests as advisory - provide warnings but don't block workflow"
    )


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """
    Check if Ollama service is available and responding.
    
    Returns:
        bool: True if Ollama is installed and the service responds, False otherwise.
    
    Note:
        This is a session-scoped fixture, so the check is performed once per test session.
    """
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


@pytest.fixture(scope="session")
def model_name() -> str:
    """
    Provide the name of the Ollama model to use for testing.
    
    Returns:
        str: The model name (default: llama3.2:1b).
    
    Note:
        Can be overridden via environment variable OLLAMA_TEST_MODEL.
    """
    return os.getenv("OLLAMA_TEST_MODEL", "llama3.2:1b")


@pytest.fixture(scope="function")
def test_output_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test outputs.
    
    Yields:
        Path: Path to a temporary directory that will be cleaned up after the test.
    
    Note:
        Directory is automatically removed after each test function completes.
    """
    with tempfile.TemporaryDirectory(prefix="ollama_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_prompt() -> str:
    """
    Provide a standard test prompt for validating model responses.
    
    Returns:
        str: A simple, deterministic prompt for testing AI functionality.
    
    Note:
        This prompt is designed to generate quick, predictable responses.
    """
    return "Respond with exactly: TEST_PASSED"


@pytest.fixture(scope="session")
def simple_query_prompt() -> str:
    """
    Provide a simple query prompt for basic functionality testing.
    
    Returns:
        str: A brief question that should generate a short response.
    """
    return "What is DevOps in one sentence?"


@pytest.fixture(scope="session")
def ollama_home() -> Path:
    """
    Provide the path to the Ollama home directory.
    
    Returns:
        Path: Path to ~/.ollama directory where Ollama stores its data.
    """
    return Path.home() / ".ollama"


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    """
    Provide the path to the Ollama binary cache directory.
    
    Returns:
        Path: Path to ~/ollama-bin directory used for caching the binary.
    """
    return Path.home() / "ollama-bin"

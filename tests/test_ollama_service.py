"""
Service health tests for Ollama installation and availability.

This module contains tests that validate the Ollama service is properly installed,
configured, and responding to requests. Tests are categorized as critical or advisory.
"""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.critical
def test_ollama_installed(ollama_available):
    """
    Verify that Ollama is installed and the binary is executable.
    
    This test checks if the 'ollama --version' command executes successfully,
    which confirms the Ollama binary is accessible in the system PATH.
    
    Args:
        ollama_available: Fixture that checks Ollama service availability.
    """
    result = subprocess.run(
        ['ollama', '--version'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    assert result.returncode == 0, (
        f"Ollama command failed with exit code {result.returncode}. "
        f"stderr: {result.stderr}"
    )
    assert 'ollama version' in result.stdout.lower(), (
        f"Unexpected version output format. "
        f"Expected 'ollama version' in output, got: {result.stdout}"
    )


@pytest.mark.critical
def test_ollama_service_responding(ollama_available):
    """
    Verify that the Ollama service is running and responding to requests.
    
    This test attempts to list available models, which requires the Ollama
    service to be actively running and accepting connections.
    
    Args:
        ollama_available: Fixture that checks Ollama service availability.
    """
    assert ollama_available, (
        "Ollama service is not available. "
        "Ensure 'ollama serve' is running in the background."
    )
    
    result = subprocess.run(
        ['ollama', 'list'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0, (
        f"Ollama service not responding. Command 'ollama list' failed with "
        f"exit code {result.returncode}. stderr: {result.stderr}"
    )


@pytest.mark.critical
def test_model_available(model_name):
    """
    Verify that the required model is downloaded and available.
    
    This test checks that the specified model (default: llama3.2:1b) appears
    in the list of available models, confirming it has been pulled successfully.
    
    Args:
        model_name: Fixture providing the name of the model to test.
    """
    result = subprocess.run(
        ['ollama', 'list'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0, (
        f"Failed to list models. Exit code: {result.returncode}, "
        f"stderr: {result.stderr}"
    )
    
    assert model_name in result.stdout, (
        f"Required model '{model_name}' not found in available models. "
        f"Run 'ollama pull {model_name}' to download it. "
        f"Available models:\n{result.stdout}"
    )


@pytest.mark.critical
def test_model_loads_successfully(model_name, sample_prompt):
    """
    Verify that the model can be loaded and process a simple prompt.
    
    This test runs a basic inference request to ensure the model loads
    correctly and can generate a response without errors.
    
    Args:
        model_name: Fixture providing the name of the model to test.
        sample_prompt: Fixture providing a standard test prompt.
    """
    result = subprocess.run(
        ['ollama', 'run', model_name, sample_prompt],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0, (
        f"Model '{model_name}' failed to process prompt. "
        f"Exit code: {result.returncode}, stderr: {result.stderr}"
    )
    
    assert len(result.stdout.strip()) > 0, (
        f"Model '{model_name}' produced empty output. "
        f"Expected a response to prompt: '{sample_prompt}'"
    )


@pytest.mark.advisory
def test_cache_directory_exists(cache_dir):
    """
    Verify that the Ollama binary cache directory exists.
    
    This is an advisory test that checks if the caching mechanism used
    in GitHub Actions is properly configured. Failure indicates caching
    may not be working but doesn't prevent Ollama from functioning.
    
    Args:
        cache_dir: Fixture providing the path to the cache directory.
    """
    assert cache_dir.exists(), (
        f"Cache directory not found at {cache_dir}. "
        f"This may indicate caching is not configured or this is the first run."
    )
    
    assert cache_dir.is_dir(), (
        f"Cache path exists but is not a directory: {cache_dir}"
    )


@pytest.mark.advisory
def test_ollama_home_exists(ollama_home):
    """
    Verify that the Ollama home directory exists.
    
    This advisory test checks for the ~/.ollama directory where Ollama
    stores models and configuration. Useful for debugging storage issues.
    
    Args:
        ollama_home: Fixture providing the path to the Ollama home directory.
    """
    assert ollama_home.exists(), (
        f"Ollama home directory not found at {ollama_home}. "
        f"This may indicate Ollama has not been initialized properly."
    )
    
    assert ollama_home.is_dir(), (
        f"Ollama home path exists but is not a directory: {ollama_home}"
    )


@pytest.mark.advisory
def test_ollama_models_directory(ollama_home):
    """
    Verify that the Ollama models directory exists and contains data.
    
    This advisory test checks if the models subdirectory exists in the
    Ollama home, which indicates models have been downloaded.
    
    Args:
        ollama_home: Fixture providing the path to the Ollama home directory.
    """
    models_dir = ollama_home / "models"
    
    if not ollama_home.exists():
        pytest.skip(f"Ollama home directory not found at {ollama_home}")
    
    assert models_dir.exists(), (
        f"Models directory not found at {models_dir}. "
        f"Models may not be downloaded or stored in a different location."
    )


@pytest.mark.critical
def test_response_quality(model_name, simple_query_prompt, test_output_dir):
    """
    Verify that the model produces coherent, non-empty responses.
    
    This test validates that the AI model not only runs but produces
    meaningful output. The response is saved to the test output directory
    for inspection if needed.
    
    Args:
        model_name: Fixture providing the name of the model to test.
        simple_query_prompt: Fixture providing a simple query prompt.
        test_output_dir: Fixture providing a temporary output directory.
    """
    result = subprocess.run(
        ['ollama', 'run', model_name, simple_query_prompt],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0, (
        f"Failed to generate response. Exit code: {result.returncode}, "
        f"stderr: {result.stderr}"
    )
    
    response = result.stdout.strip()
    
    # Save response for debugging
    output_file = test_output_dir / "test_response.txt"
    output_file.write_text(response)
    
    assert len(response) > 10, (
        f"Response too short (< 10 characters). "
        f"Expected meaningful output, got: '{response}'. "
        f"Full response saved to: {output_file}"
    )
    
    # Check for common error patterns
    error_indicators = ['error', 'failed', 'exception', 'traceback']
    response_lower = response.lower()
    
    for indicator in error_indicators:
        assert indicator not in response_lower, (
            f"Response contains error indicator '{indicator}'. "
            f"Response: {response}"
        )

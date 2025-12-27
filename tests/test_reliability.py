"""
Reliability Tests for Ollama AI Workflow
Tests failure modes and graceful error handling
"""

import subprocess
import pytest
import time
import sys


class TestReliability:
    """Test suite for reliability and error handling"""

    def test_handles_invalid_model(self):
        """Test graceful failure with non-existent model"""
        result = subprocess.run(
            ["ollama", "run", "nonexistent-model-xyz", "hello"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should return non-zero exit code for invalid model
        assert result.returncode != 0, "Expected non-zero exit code for invalid model"

        # Error message should be in stderr
        assert result.stderr, "Expected error message in stderr"

        # Error should mention the model or indicate it's not found
        error_lower = result.stderr.lower()
        assert any(word in error_lower for word in ["not found", "error", "failed", "pull", "unknown"]), \
            f"Error message should indicate model issue: {result.stderr}"

    def test_handles_empty_prompt(self):
        """Test appropriate response to empty input"""
        result = subprocess.run(
            ["ollama", "run", "llama3.2:1b", ""],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Empty prompt should either succeed with empty/minimal response
        # or fail gracefully - either is acceptable
        if result.returncode == 0:
            # If it succeeds, output should exist (even if minimal)
            assert isinstance(result.stdout, str), "Output should be a string"
        else:
            # If it fails, should have meaningful error
            assert result.stderr or result.stdout, "Should have some output on failure"

    def test_handles_timeout(self):
        """Test behavior when query times out"""
        timed_out = False
        start_time = time.time()

        try:
            # Use a very short timeout to force timeout condition
            result = subprocess.run(
                ["ollama", "run", "llama3.2:1b", "Write a 10000 word essay about the universe"],
                capture_output=True,
                text=True,
                timeout=5  # Very short timeout
            )
        except subprocess.TimeoutExpired:
            timed_out = True

        elapsed = time.time() - start_time

        # Either timeout occurred OR completed within reasonable time
        assert timed_out or elapsed < 60, \
            "Query should either timeout or complete within reasonable time"

        if timed_out:
            # Verify timeout happened around expected time
            assert elapsed < 10, "Timeout should trigger within expected window"

    def test_partial_failure_recovery(self):
        """Test that workflow continues after non-critical failure"""
        results = []

        # Simulate a workflow with multiple steps, some failing
        test_cases = [
            ("invalid-model-name", "test", False),  # Expected to fail
            ("llama3.2:1b", "Say hello", True),        # Expected to succeed
            ("another-fake-model", "test", False),   # Expected to fail
        ]

        for model, prompt, should_succeed in test_cases:
            try:
                result = subprocess.run(
                    ["ollama", "run", model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                results.append({
                    "model": model,
                    "success": result.returncode == 0,
                    "expected": should_succeed
                })
            except subprocess.TimeoutExpired:
                results.append({
                    "model": model,
                    "success": False,
                    "expected": should_succeed,
                    "timeout": True
                })
            except Exception as e:
                results.append({
                    "model": model,
                    "success": False,
                    "expected": should_succeed,
                    "error": str(e)
                })

        # All test cases should have been attempted (no early exit)
        assert len(results) == len(test_cases), \
            f"All {len(test_cases)} tests should run, got {len(results)}"

        # Verify expected outcomes
        for result in results:
            if result["expected"]:
                assert result["success"], \
                    f"Model {result['model']} should have succeeded"

    def test_error_messages_helpful(self):
        """Test that error output contains actionable information"""
        result = subprocess.run(
            ["ollama", "run", "this-model-does-not-exist-12345", "test"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should fail
        assert result.returncode != 0, "Invalid model should fail"

        # Combine all output for checking
        all_output = (result.stderr + result.stdout).lower()

        # Error message should contain helpful information
        helpful_indicators = [
            "not found",
            "error",
            "pull",
            "try",
            "model",
            "failed",
            "unable",
            "does not exist"
        ]

        has_helpful_info = any(indicator in all_output for indicator in helpful_indicators)
        assert has_helpful_info, \
            f"Error should contain actionable info. Got: {result.stderr or result.stdout}"


class TestFailureModes:
    """Verify the system fails gracefully"""

    def test_invalid_model_returns_nonzero(self):
        """Invalid model names return non-zero exit code"""
        result = subprocess.run(
            ["ollama", "run", "fake_model_xyz_123", "test"],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode != 0, \
            "Invalid model should return non-zero exit code"

    def test_error_messages_captured_in_stderr(self):
        """Error messages are captured in stderr"""
        result = subprocess.run(
            ["ollama", "run", "nonexistent_model_abc", "test"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Error info should be in stderr (or stdout for some error types)
        has_error_output = bool(result.stderr) or (
            result.returncode != 0 and bool(result.stdout)
        )

        assert has_error_output, \
            "Error information should be captured in output streams"

    def test_partial_failures_dont_crash_suite(self):
        """Partial failures don't crash the entire test suite"""
        failed_count = 0
        success_count = 0
        total_attempted = 0

        test_prompts = [
            ("llama3.2:1b", "Hi"),           # Should work
            ("bad-model-1", "test"),       # Should fail
            ("llama3.2:1b", "Hello"),        # Should work
            ("bad-model-2", "test"),       # Should fail
            ("llama3.2:1b", "Bye"),          # Should work
        ]

        for model, prompt in test_prompts:
            total_attempted += 1
            try:
                result = subprocess.run(
                    ["ollama", "run", model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1

        # All tests should have been attempted
        assert total_attempted == len(test_prompts), \
            "All tests should be attempted regardless of failures"

        # Should have both successes and failures (mixed results)
        assert success_count > 0, "Some valid model tests should succeed"
        assert failed_count > 0, "Some invalid model tests should fail"


class TestOllamaServiceReliability:
    """Test Ollama service availability and reliability"""

    def test_ollama_service_running(self):
        """Verify Ollama service is accessible"""
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"Ollama service should be running. Error: {result.stderr}"

    def test_model_list_available(self):
        """Verify we can list available models"""
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, "Should be able to list models"
        # Output should be a valid response (could be empty if no models)
        assert isinstance(result.stdout, str), "Output should be a string"

    def test_concurrent_request_handling(self):
        """Test handling of rapid sequential requests"""
        results = []

        # Make several rapid requests
        for i in range(3):
            result = subprocess.run(
                ["ollama", "run", "llama3.2:1b", f"Count to {i+1}"],
                capture_output=True,
                text=True,
                timeout=60
            )
            results.append(result.returncode)

        # All requests should complete (success or graceful failure)
        assert len(results) == 3, "All requests should complete"

        # At least some should succeed
        successes = sum(1 for r in results if r == 0)
        assert successes > 0, "At least some concurrent requests should succeed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

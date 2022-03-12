"""Tests for fi_instrumentation.settings module."""

import logging
import os
import pytest
import uuid
from unittest.mock import patch, MagicMock
from fi_instrumentation.settings import (
    get_env_collector_endpoint,
    get_env_grpc_collector_endpoint,
    get_env_project_name,
    get_env_project_version_name,
    get_env_fi_auth_header,
    parse_env_headers,
    UuidIdGenerator,
    get_custom_eval_template,
)


class TestEnvironmentVariableGetters:
    """Test environment variable getter functions."""

    def test_collector_urls_point_at_a_closed_local_port_in_every_test(self, request):
        """conftest.py's autouse fixture keeps every test off the real collector (no clean_env here)."""
        assert "_collector_at_a_closed_local_port" in request.fixturenames
        assert os.environ["FI_BASE_URL"] == "http://127.0.0.1:9"
        assert os.environ["FI_GRPC_URL"] == "http://127.0.0.1:9"

    def test_get_env_collector_endpoint_default(self, clean_env, monkeypatch):
        """Test default collector endpoint."""
        monkeypatch.delenv("FI_BASE_URL", raising=False)
        result = get_env_collector_endpoint()
        assert result == "https://api.futureagi.com"

    def test_get_env_collector_endpoint_custom(self, clean_env):
        """Test custom collector endpoint from environment."""
        os.environ["FI_BASE_URL"] = "https://custom.example.com"
        result = get_env_collector_endpoint()
        assert result == "https://custom.example.com"

    def test_get_env_grpc_collector_endpoint_default(self, clean_env, monkeypatch):
        """Test default gRPC collector endpoint."""
        monkeypatch.delenv("FI_GRPC_URL", raising=False)
        result = get_env_grpc_collector_endpoint()
        assert result == "https://grpc.futureagi.com"

    def test_get_env_grpc_collector_endpoint_custom(self, clean_env):
        """Test custom gRPC collector endpoint from environment."""
        os.environ["FI_GRPC_URL"] = "https://custom-grpc.example.com:50051"
        result = get_env_grpc_collector_endpoint()
        assert result == "https://custom-grpc.example.com:50051"

    def test_get_env_project_name_default(self, clean_env):
        """Test default project name."""
        result = get_env_project_name()
        assert result == "DEFAULT_PROJECT_NAME"

    def test_get_env_project_name_custom(self, clean_env):
        """Test custom project name from environment."""
        os.environ["FI_PROJECT_NAME"] = "my_custom_project"
        result = get_env_project_name()
        assert result == "my_custom_project"

    def test_get_env_project_version_name_default(self, clean_env):
        """Test default project version name."""
        result = get_env_project_version_name()
        assert result == "DEFAULT_PROJECT_VERSION_NAME"

    def test_get_env_project_version_name_custom(self, clean_env):
        """Test custom project version name from environment."""
        os.environ["FI_PROJECT_VERSION_NAME"] = "v2.0.0"
        result = get_env_project_version_name()
        assert result == "v2.0.0"


class TestAuthHeaders:
    """Test authentication header functionality."""

    def test_get_env_fi_auth_header_both_keys(self, clean_env):
        """Test auth header when both keys are present."""
        os.environ["FI_API_KEY"] = "test_api_key"
        os.environ["FI_SECRET_KEY"] = "test_secret_key"
        
        result = get_env_fi_auth_header()
        
        assert result == {
            "X-Api-Key": "test_api_key",
            "X-Secret-Key": "test_secret_key"
        }

    def test_get_env_fi_auth_header_missing_api_key(self, clean_env):
        """Test auth header when API key is missing."""
        os.environ["FI_SECRET_KEY"] = "test_secret_key"
        
        result = get_env_fi_auth_header()
        assert result is None

    def test_get_env_fi_auth_header_missing_secret_key(self, clean_env):
        """Test auth header when secret key is missing."""
        os.environ["FI_API_KEY"] = "test_api_key"
        
        result = get_env_fi_auth_header()
        assert result is None

    def test_get_env_fi_auth_header_no_keys(self, clean_env):
        """Test auth header when no keys are present."""
        result = get_env_fi_auth_header()
        assert result is None


class TestHeaderParsing:
    """Test header parsing functionality."""

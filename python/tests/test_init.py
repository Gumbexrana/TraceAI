"""Tests for fi_instrumentation.__init__ module (public API)."""

import pytest
from unittest.mock import patch, MagicMock

# Test that all expected symbols are properly exported
def test_public_api_exports():
    """Test that all expected symbols are available in the public API."""
    import fi_instrumentation
    
    # Context and attributes
    assert hasattr(fi_instrumentation, 'get_attributes_from_context')
    assert hasattr(fi_instrumentation, 'using_attributes')
    assert hasattr(fi_instrumentation, 'using_metadata')
    assert hasattr(fi_instrumentation, 'using_prompt_template')
    assert hasattr(fi_instrumentation, 'using_session')
    assert hasattr(fi_instrumentation, 'using_tags')
    assert hasattr(fi_instrumentation, 'using_user')
    
    # Helpers and config
    assert hasattr(fi_instrumentation, 'safe_json_dumps')
    assert hasattr(fi_instrumentation, 'suppress_tracing')
    assert hasattr(fi_instrumentation, 'TraceConfig')
    assert hasattr(fi_instrumentation, 'FITracer')
    assert hasattr(fi_instrumentation, 'REDACTED_VALUE')
    
    # OpenTelemetry components
    assert hasattr(fi_instrumentation, 'TracerProvider')
    assert hasattr(fi_instrumentation, 'SimpleSpanProcessor')
    assert hasattr(fi_instrumentation, 'BatchSpanProcessor')
    assert hasattr(fi_instrumentation, 'HTTPSpanExporter')
    assert hasattr(fi_instrumentation, 'Resource')
    assert hasattr(fi_instrumentation, 'PROJECT_NAME')
    assert hasattr(fi_instrumentation, 'PROJECT_TYPE')
    assert hasattr(fi_instrumentation, 'PROJECT_VERSION_NAME')
    assert hasattr(fi_instrumentation, 'Transport')
    assert hasattr(fi_instrumentation, 'register')


def test_context_managers_import():
    """Test that context managers can be imported and used."""
    from fi_instrumentation import using_session, using_user, using_metadata
    
    # Test that they can be instantiated
    session_mgr = using_session("test-session")
    user_mgr = using_user("test-user")
    metadata_mgr = using_metadata({"key": "value"})
    
    assert session_mgr is not None
    assert user_mgr is not None
    assert metadata_mgr is not None


def test_tracer_import():
    """Test that FITracer can be imported and instantiated."""
    from fi_instrumentation import FITracer, TraceConfig
    from unittest.mock import MagicMock
    from opentelemetry.trace import Tracer
    
    mock_tracer = MagicMock(spec=Tracer)
    config = TraceConfig()
    fi_tracer = FITracer(mock_tracer, config)
    
    assert fi_tracer is not None
    assert hasattr(fi_tracer, 'agent')
    assert hasattr(fi_tracer, 'chain')
    assert hasattr(fi_tracer, 'tool')


def test_otel_components_import():
    """Test that OpenTelemetry components can be imported."""
    from fi_instrumentation import (
        TracerProvider,
        SimpleSpanProcessor,
        BatchSpanProcessor,
        HTTPSpanExporter,
        Transport,
        register
    )
    
    # Test that classes/functions are available
    assert TracerProvider is not None
    assert SimpleSpanProcessor is not None
    assert BatchSpanProcessor is not None
    assert HTTPSpanExporter is not None
    assert Transport is not None
    assert register is not None


def test_constants_import():
    """Test that constants can be imported."""
    from fi_instrumentation import (
        PROJECT_NAME,
        PROJECT_TYPE,
        PROJECT_VERSION_NAME,
        REDACTED_VALUE
    )
    
    assert PROJECT_NAME == "project_name"
    assert PROJECT_TYPE == "project_type"
    assert PROJECT_VERSION_NAME == "project_version_name"
    assert REDACTED_VALUE == "__REDACTED__"


def test_helper_functions_import():
    """Test that helper functions can be imported and used."""
    from fi_instrumentation import safe_json_dumps, get_attributes_from_context
    
    # Test safe_json_dumps
    test_data = {"key": "value", "number": 123}
    json_str = safe_json_dumps(test_data)
    assert isinstance(json_str, str)
    assert "key" in json_str
    assert "value" in json_str

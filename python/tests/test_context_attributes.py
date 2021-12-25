"""Tests for fi_instrumentation.instrumentation.context_attributes module."""

import pytest
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
from opentelemetry.context import get_current, get_value

from fi_instrumentation.instrumentation.context_attributes import (
    using_session,
    using_user,
    using_metadata,
    using_tags,
    using_prompt_template,
    using_attributes,
    get_attributes_from_context,
    CONTEXT_ATTRIBUTES,
)
from fi_instrumentation.fi_types import SimulatorAttributes, SpanAttributes


class TestUsingSession:
    """Test using_session context manager."""

    def test_using_session_sync_context(self):
        """Test using_session as synchronous context manager."""
        session_id = "test-session-123"
        
        # Initially, session ID should not be in context
        initial_value = get_value(SpanAttributes.SESSION_ID)
        assert initial_value is None
        
        with using_session(session_id):
            # Inside context, session ID should be set
            current_value = get_value(SpanAttributes.SESSION_ID)
            assert current_value == session_id
        
        # After context, session ID should be cleared
        final_value = get_value(SpanAttributes.SESSION_ID)
        assert final_value is None

    @pytest.mark.asyncio
    async def test_using_session_async_context(self):
        """Test using_session as asynchronous context manager."""
        session_id = "test-session-async-456"
        
        # Initially, session ID should not be in context
        initial_value = get_value(SpanAttributes.SESSION_ID)
        assert initial_value is None
        
        async with using_session(session_id):
            # Inside context, session ID should be set
            current_value = get_value(SpanAttributes.SESSION_ID)
            assert current_value == session_id
        
        # After context, session ID should be cleared
        final_value = get_value(SpanAttributes.SESSION_ID)
        assert final_value is None

    def test_using_session_nested(self):
        """Test nested using_session contexts."""
        session_id1 = "session-1"
        session_id2 = "session-2"
        
        with using_session(session_id1):
            assert get_value(SpanAttributes.SESSION_ID) == session_id1
            
            with using_session(session_id2):
                assert get_value(SpanAttributes.SESSION_ID) == session_id2
            
            # Should revert to outer session
            assert get_value(SpanAttributes.SESSION_ID) == session_id1
        
        # Should be cleared after all contexts
        assert get_value(SpanAttributes.SESSION_ID) is None

    def test_using_session_exception_handling(self):
        """Test using_session properly handles exceptions."""
        session_id = "test-session-exception"
        
        try:
            with using_session(session_id):
                assert get_value(SpanAttributes.SESSION_ID) == session_id
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still clear session ID after exception
        final_value = get_value(SpanAttributes.SESSION_ID)
        assert final_value is None


class TestUsingUser:
    """Test using_user context manager."""

    def test_using_user_sync_context(self):
        """Test using_user as synchronous context manager."""
        user_id = "user-123"
        
        with using_user(user_id):
            current_value = get_value(SpanAttributes.USER_ID)
            assert current_value == user_id
        
        final_value = get_value(SpanAttributes.USER_ID)
        assert final_value is None

    @pytest.mark.asyncio
    async def test_using_user_async_context(self):
        """Test using_user as asynchronous context manager."""
        user_id = "user-async-456"
        
        async with using_user(user_id):
            current_value = get_value(SpanAttributes.USER_ID)
            assert current_value == user_id
        
        final_value = get_value(SpanAttributes.USER_ID)
        assert final_value is None

    def test_using_user_nested(self):
        """Test nested using_user contexts."""
        user_id1 = "user-1"
        user_id2 = "user-2"
        
        with using_user(user_id1):
            assert get_value(SpanAttributes.USER_ID) == user_id1
            
            with using_user(user_id2):
                assert get_value(SpanAttributes.USER_ID) == user_id2
            
            assert get_value(SpanAttributes.USER_ID) == user_id1
        
        assert get_value(SpanAttributes.USER_ID) is None


class TestUsingMetadata:
    """Test using_metadata context manager."""

    def test_using_metadata_sync_context(self):
        """Test using_metadata as synchronous context manager."""
        metadata = {"key1": "value1", "key2": "value2"}
        
        with using_metadata(metadata):
            current_value = get_value(SpanAttributes.METADATA)
            # Should be JSON serialized
            assert current_value == '{"key1": "value1", "key2": "value2"}'
        
        final_value = get_value(SpanAttributes.METADATA)
        assert final_value is None

    @pytest.mark.asyncio
    async def test_using_metadata_async_context(self):
        """Test using_metadata as asynchronous context manager."""
        metadata = {"async_key": "async_value", "number": 42}
        
        async with using_metadata(metadata):
            current_value = get_value(SpanAttributes.METADATA)
            assert '{"async_key": "async_value", "number": 42}' in current_value or \
                   '{"number": 42, "async_key": "async_value"}' in current_value
        
        final_value = get_value(SpanAttributes.METADATA)
        assert final_value is None

    def test_using_metadata_complex_data(self):
        """Test using_metadata with complex data structures."""
        metadata = {
            "string": "value",
            "number": 123,
            "boolean": True,
            "list": [1, 2, 3],
            "nested": {"inner": "data"}
        }
        
        with using_metadata(metadata):
            current_value = get_value(SpanAttributes.METADATA)
            assert current_value is not None
            # Should be valid JSON

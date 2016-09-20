"""Tests for AutogenInstrumentor class."""

import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestAutoGenInstrumentor:
    """Tests for AutogenInstrumentor class."""

    @pytest.fixture
    def mock_autogen_v02(self):
        """Create mock autogen v0.2 module."""
        mock_module = MagicMock()
        mock_module.ConversableAgent = MagicMock()
        mock_module.ConversableAgent.generate_reply = MagicMock()
        mock_module.ConversableAgent.initiate_chat = MagicMock()
        mock_module.ConversableAgent.execute_function = MagicMock()
        return mock_module

    @pytest.fixture
    def mock_autogen_v04(self):
        """Create mock autogen v0.4 modules."""
        # Agents module
        agents_module = MagicMock()

        # Create BaseChatAgent class
        BaseChatAgent = type("BaseChatAgent", (), {})
        BaseChatAgent.on_messages = MagicMock()
        agents_module.BaseChatAgent = BaseChatAgent

        # Create AssistantAgent class
        AssistantAgent = type("AssistantAgent", (BaseChatAgent,), {})
        agents_module.AssistantAgent = AssistantAgent

        # Teams module
        teams_module = MagicMock()

        # Create team classes
        BaseGroupChat = type("BaseGroupChat", (), {})
        BaseGroupChat.run = MagicMock()
        BaseGroupChat.run_stream = MagicMock()
        teams_module.BaseGroupChat = BaseGroupChat

        RoundRobinGroupChat = type("RoundRobinGroupChat", (BaseGroupChat,), {})
        RoundRobinGroupChat.run = MagicMock()
        RoundRobinGroupChat.run_stream = MagicMock()
        teams_module.RoundRobinGroupChat = RoundRobinGroupChat

        SelectorGroupChat = type("SelectorGroupChat", (BaseGroupChat,), {})
        SelectorGroupChat.run = MagicMock()
        SelectorGroupChat.run_stream = MagicMock()
        teams_module.SelectorGroupChat = SelectorGroupChat

        return {
            "autogen_agentchat.agents": agents_module,
            "autogen_agentchat.teams": teams_module,
        }

    def test_instrumentor_initialization(self):
        """Test instrumentor initialization."""
        with patch.dict(sys.modules, {"autogen": MagicMock()}):
            from traceai_autogen import AutogenInstrumentor

            instrumentor = AutogenInstrumentor()

            assert instrumentor._original_generate is None
            assert instrumentor._original_initiate_chat is None
            assert instrumentor._original_execute_function is None
            assert instrumentor._v04_original_on_messages == {}
            assert instrumentor._v04_original_team_run == {}
            assert instrumentor._v04_original_team_run_stream == {}
            assert instrumentor.tracer is None

    def test_instrument_v02_only(self, mock_autogen_v02):
        """Test instrumenting only v0.2."""
        modules_dict = {
            "autogen": mock_autogen_v02,
        }

        with patch.dict(sys.modules, modules_dict, clear=False):
            # Re-import to get fresh module with patches
            import importlib
            import traceai_autogen
            importlib.reload(traceai_autogen)

            from traceai_autogen import AutogenInstrumentor

            mock_tracer_provider = MagicMock()
            mock_tracer = MagicMock()
            mock_tracer_provider.get_tracer = MagicMock(return_value=mock_tracer)

            with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_tracer_provider):
                with patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):
                    instrumentor = AutogenInstrumentor()
                    instrumentor.instrument()

                    # Verify v0.2 methods were stored
                    assert instrumentor._original_generate is not None
                    assert instrumentor._original_initiate_chat is not None
                    assert instrumentor._original_execute_function is not None

    def test_uninstrument_restores_v02(self, mock_autogen_v02):
        """Test uninstrumenting restores v0.2 methods."""
        original_generate = MagicMock()
        original_initiate_chat = MagicMock()
        original_execute_function = MagicMock()

        mock_autogen_v02.ConversableAgent.generate_reply = original_generate
        mock_autogen_v02.ConversableAgent.initiate_chat = original_initiate_chat
        mock_autogen_v02.ConversableAgent.execute_function = original_execute_function

        modules_dict = {
            "autogen": mock_autogen_v02,
        }

        with patch.dict(sys.modules, modules_dict, clear=False):
            import importlib
            import traceai_autogen
            importlib.reload(traceai_autogen)

            from traceai_autogen import AutogenInstrumentor

            mock_tracer_provider = MagicMock()
            mock_tracer = MagicMock()

            with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_tracer_provider):
                with patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):
                    instrumentor = AutogenInstrumentor()
                    instrumentor.instrument()

                    # Methods should be wrapped now
                    assert mock_autogen_v02.ConversableAgent.generate_reply != original_generate

                    # Uninstrument
                    instrumentor.uninstrument()

                    # Originals should be restored
                    assert mock_autogen_v02.ConversableAgent.generate_reply == original_generate
                    assert mock_autogen_v02.ConversableAgent.initiate_chat == original_initiate_chat
                    assert mock_autogen_v02.ConversableAgent.execute_function == original_execute_function

    def test_safe_json_dumps(self, mock_autogen_v02):
        """Test _safe_json_dumps method."""
        with patch.dict(sys.modules, {"autogen": mock_autogen_v02}):
            from traceai_autogen import AutogenInstrumentor

            instrumentor = AutogenInstrumentor()

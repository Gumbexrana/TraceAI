import inspect
import logging
import base64
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Tuple, TypeVar, cast

from google.genai.types import Content, FunctionCall, FunctionResponse, Part, UserContent
from opentelemetry.util.types import AttributeValue

from fi_instrumentation import safe_json_dumps
from traceai_google_genai._utils import (
    _as_input_attributes,
    _io_value_and_type,
)
from fi_instrumentation.fi_types import (
    MessageAttributes,
    MessageContentAttributes,
    FiLLMProviderValues,
    FiSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

__all__ = ("_RequestAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _RequestAttributesExtractor:
    __slots__ = ()

    def get_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.GEN_AI_SPAN_KIND, FiSpanKindValues.LLM.value
        yield SpanAttributes.GEN_AI_PROVIDER_NAME, FiLLMProviderValues.GOOGLE.value
        try:
            yield from _as_input_attributes(
                _io_value_and_type(request_parameters),
            )
        except Exception:
            logger.exception(
                f"Failed to get input attributes from request parameters of "
                f"type {type(request_parameters)}"
            )

        # Extract tools as high-priority attributes (avoid 128 attribute limit dropping them)
        if isinstance(request_parameters, Mapping):
            if config := request_parameters.get("config", None):
                yield from self._get_tools_from_config(config)

    def get_extra_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # Start an index for the messages since we want to start with system instruction
        input_messages_index = 0
        if not isinstance(request_parameters, Mapping):
            return

        request_params_dict = dict(request_parameters)
        request_params_dict.pop("contents", None)  # Remove LLM input contents
        if config := request_params_dict.get("config", None):
            # config can either be a TypedDict or a pydantic object so we need to handle both cases
            if isinstance(config, dict):
                config_json = safe_json_dumps(config)
            else:
                config_json = self._serialize_config_safely(config)
            yield (
                SpanAttributes.GEN_AI_REQUEST_PARAMETERS,
                config_json,
            )

            # We push the system instruction to the first message for replay and consistency
            system_instruction = getattr(config, "system_instruction", None)
            if system_instruction:
                yield (
                    f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_CONTENT}",
                    system_instruction,
                )
                yield (
                    f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_ROLE}",
                    "system",
                )
                input_messages_index += 1

            # Tools are now extracted in get_attributes_from_request for higher priority

        if input_contents := request_parameters.get("contents"):
            if isinstance(input_contents, list):
                for input_content in input_contents:
                    for attr, value in self._get_attributes_from_message_param(input_content):
                        yield (
                            f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                            value,
                        )
                    # Move on to the next message
                    input_messages_index += 1
            else:
                for attr, value in self._get_attributes_from_message_param(input_contents):
                    # Default to index 0 for a single message
                    yield (
                        f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                        value,
                    )

    def _serialize_config_safely(self, config: Any) -> str:
        """
        Serialize a config object to JSON, handling Pydantic model classes
        that can't be directly serialized.
        """
        # Create a copy with serializable values - excluding tools which may contain a python
        # function object which will fail to serialize correctly later on
        config_dict = config.model_dump(exclude_none=True, exclude={"tools"})

        # Handle response_schema if it's a Pydantic model class
        if "response_schema" in config_dict:
            response_schema = config_dict["response_schema"]
            if hasattr(response_schema, "__pydantic_core_schema__"):
                # It's a Pydantic model class, convert to schema name or string representation
                config_dict["response_schema"] = getattr(
                    response_schema, "__name__", str(response_schema)
                )
            elif hasattr(response_schema, "model_json_schema"):
                # It's a Pydantic model instance, use its schema
                config_dict["response_schema"] = response_schema.model_json_schema()

        return safe_json_dumps(config_dict)

    def _get_tools_from_config(self, config: Any) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract tools from the GenerateContentConfig object."""
        if not hasattr(config, "tools") or not config.tools:
            return

        tools = config.tools
        if not isinstance(tools, list):
            return

        tool_index = 0  # Track across all function declarations

        for tool in tools:
            try:
                # Handle different types of tools
                if callable(tool):
                    # Python function for automatic function calling
                    tool_dict = self._convert_automatic_function_to_schema(tool)
                elif hasattr(tool, "model_dump"):
                    # Pydantic model
                    tool_dict = tool.model_dump(exclude_none=True)
                elif hasattr(tool, "__dict__"):
                    # Regular object with attributes
                    tool_dict = self._convert_tool_to_dict(tool)
                else:
                    # Already a dict or other serializable format
                    tool_dict = tool

                # Extract function declarations and create separate tool entries
                if isinstance(tool_dict, dict) and "function_declarations" in tool_dict:
                    function_declarations = tool_dict["function_declarations"]
                    if isinstance(function_declarations, list):
                        for func_decl in function_declarations:
                            # Convert Google GenAI format to flattened format
                            flattened_format = self._convert_to_flattened_format(func_decl)
                            yield (
                                f"{SpanAttributes.GEN_AI_TOOL_DEFINITIONS}.{tool_index}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                                safe_json_dumps(flattened_format),
                            )
                            tool_index += 1
                else:
                    # Tool doesn't have function_declarations, use as-is
                    yield (
                        f"{SpanAttributes.GEN_AI_TOOL_DEFINITIONS}.{tool_index}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                        safe_json_dumps(tool_dict),
                    )
                    tool_index += 1

            except Exception:
                logger.exception(f"Failed to serialize tool: {tool}")

    def _convert_to_flattened_format(self, func_decl: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Google GenAI function declaration to flattened format."""
        flattened_tool = {}

        # Copy name and description directly
        if "name" in func_decl:
            flattened_tool["name"] = func_decl["name"]
        if "description" in func_decl:
            flattened_tool["description"] = func_decl["description"]

        # flatten parameters
        if "parameters" in func_decl:
            flattened_tool["parameters"] = func_decl["parameters"]

        return flattened_tool

    def _convert_automatic_function_to_schema(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """Convert a Python function to a tool schema for automatic function calling."""
        import inspect
        from typing import get_type_hints

        # Get function metadata - cast to tell mypy that functions have these attributes
        func_name = cast(Any, func).__name__
        func_doc = cast(Any, func).__doc__ or ""

        # Get function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Build parameters schema

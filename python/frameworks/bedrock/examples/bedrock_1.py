import json
import os

import boto3
from botocore.exceptions import ClientError
from fi_instrumentation import register
from fi_instrumentation.fi_types import (
    EvalName,
    EvalSpanKind,
    EvalTag,
    EvalTagType,
    ProjectType,
)
from traceai_bedrock import BedrockInstrumentor

# Configure trace provider with custom evaluation tags
eval_tags = [
    EvalTag(
        eval_name=EvalName.DETERMINISTIC_EVALS,
        value=EvalSpanKind.TOOL,
        type=EvalTagType.OBSERVATION_SPAN,
        config={
            "multi_choice": False,
            "choices": ["Yes", "No"],
            "rule_prompt": "Evaluate if the response is correct",
        },
        custom_eval_name="<custom_eval_name>",
    )
]

# Configure trace provider with custom evaluation tags
trace_provider = register(
    project_type=ProjectType.EXPERIMENT,
    eval_tags=eval_tags,
    project_name="FUTURE_AGI",
    project_version_name="v1",
)

# Instrument Bedrock client
BedrockInstrumentor().instrument(tracer_provider=trace_provider)

client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def converse_with_claude():
    system_prompt = [{"text": "You are an expert at creating music playlists"}]
    messages = [
        {
            "role": "user",
            "content": [{"text": "Hello, how are you?"}, {"text": "What's your name?"}],
        }
    ]
    inference_config = {"maxTokens": 1024, "temperature": 0.0}

    response = client.converse(
        modelId=os.getenv("AWS_BEDROCK_ANTHROPIC_ARN_ID"),
        system=system_prompt,
        messages=messages,
        inferenceConfig=inference_config,
    )
    out = response["output"]["message"]
    messages.append(out)


def invoke_claude_native():

    model_id = os.getenv("AWS_BEDROCK_ANTHROPIC_ARN_ID")

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, how are you?"},
                    {"type": "text", "text": "What's your name?"},
                ],
            }
        ],
    }

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(request_body))
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
    except ClientError as e:
        return f"Error: {str(e)}"


def stream_claude_native():

    model_id = os.getenv("AWS_BEDROCK_ANTHROPIC_ARN_ID")

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, how are you?"},
                    {"type": "text", "text": "What's your name?"},
                ],
            }
        ],
    }

    try:
        response = client.invoke_model_with_response_stream(
            modelId=model_id, body=json.dumps(request_body)
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                print(chunk["delta"].get("text", ""), end="")
    except ClientError as e:
        print(f"Error: {str(e)}")


def converse_with_tools():

    model_id = os.getenv("AWS_BEDROCK_ANTHROPIC_ARN_ID")

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state",
                                }
                            },
                            "required": ["location"],
                        }
                    },
                }
            }
        ]
    }

    conversation = [
        {"role": "user", "content": [{"text": "What's the weather like in New York?"}]}
    ]


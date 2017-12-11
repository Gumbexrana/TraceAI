# TraceAI Google ADK Instrumentation

[![pypi](https://badge.fury.io/py/traceai-google-adk.svg)](https://pypi.org/project/traceai-google-adk/)

Python auto-instrumentation library for Google ADK.


## Quickstart

In this example we will instrument a small program that uses Gemini and observe the traces in Future AGI Dashboard

Install packages.

```shell
pip install traceai-google-adk
```

In a python file, set up the `GoogleADKInstrumentor` and configure the tracer to send traces to Observe.

```python
import asyncio

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from traceai_google_adk import GoogleADKInstrumentor

tracer_provider = register(
    project_name="test_project",
    project_type=ProjectType.OBSERVE,
)


GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:

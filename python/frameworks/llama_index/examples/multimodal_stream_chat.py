from fi_instrumentation import register
from fi_instrumentation.fi_types import (
    EvalName,
    EvalSpanKind,
    EvalTag,
    EvalTagType,
    ProjectType,
)
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.openai.utils import (
    generate_openai_multi_modal_chat_message,
)
from traceai_llamaindex import LlamaIndexInstrumentor

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

# Initialize the Llama Index instrumentor
LlamaIndexInstrumentor().instrument(tracer_provider=trace_provider)


IMAGE_URLS = [
    # "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg",
    # "https://www.sportsnet.ca/wp-content/uploads/2023/11/CP1688996471-1040x572.jpg",
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",  # noqa: E501
    # "https://www.cleverfiles.com/howto/wp-content/uploads/2018/03/minion.jpg",
]

if __name__ == "__main__":
    image_documents = load_image_urls(IMAGE_URLS)

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o",
    )

    chat_msg_1 = generate_openai_multi_modal_chat_message(
        prompt="Describe the images as an alternative text",

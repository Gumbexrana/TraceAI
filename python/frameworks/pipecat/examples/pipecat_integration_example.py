import os

from dotenv import load_dotenv

load_dotenv(override=True)

from dotenv import load_dotenv
from fi_instrumentation import Transport, register
from fi_instrumentation.fi_types import ProjectType
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from traceai_pipecat import enable_fi_attribute_mapping

trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name=f"Pipecat Voice App GRPC Batch",
    transport=Transport.GRPC,
    batch=True,
    set_global_tracer_provider=True,
)

try:
    success = enable_fi_attribute_mapping(transport=Transport.GRPC)
    if success:
        print("✅ Successfully installed attribute mapping with auto-detection")
    else:
        print("❌ Failed to install attribute mapping with auto-detection")
except Exception as e:
    print(f"❌ Error installing attribute mapping: {e}")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )


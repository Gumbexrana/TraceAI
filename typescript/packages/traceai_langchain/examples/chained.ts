import { register, ProjectType } from "@traceai/fi-core";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { LangChainInstrumentation } from "../src";
import "./instrumentationchat";
import "dotenv/config";
import { randomUUID } from 'crypto';
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { createOpenAIToolsAgent, AgentExecutor } from "langchain/agents";
import { Calculator } from "@langchain/community/tools/calculator";
import { DynamicTool } from "@langchain/core/tools";
import { RunnableSequence } from "@langchain/core/runnables";
import { pull } from "langchain/hub";

// Enable OpenTelemetry internal diagnostics
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

const main = async () => {
  // FI Telemetry Credentials
  const fiApiKey = process.env.FI_API_KEY;
  const fiSecretKey = process.env.FI_SECRET_KEY;
  if (!fiApiKey || !fiSecretKey) {
    console.error(
      "FI_API_KEY and FI_SECRET_KEY environment variables must be set for telemetry.",
    );
    process.exit(1);
  }

  // Log environment variables for debugging
  console.log("Environment variables:");
  console.log("FI_BASE_URL:", process.env.FI_BASE_URL);
  console.log("FI_COLLECTOR_ENDPOINT:", process.env.FI_COLLECTOR_ENDPOINT);
  console.log("FI_API_KEY:", fiApiKey);
  console.log("FI_SECRET_KEY:", fiSecretKey);

  // 1. Register FI Core TracerProvider (sets up exporter)
  const tracerProvider = register({
    projectName: "langchain-chained-agent",
    projectType: ProjectType.OBSERVE,
    sessionName: "langchain"
  });

  // 2. Register LangChain Instrumentation
  registerInstrumentations({
    tracerProvider: tracerProvider,
    instrumentations: [new LangChainInstrumentation()],
  });

  // 3. Create a simple chain
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
  });

  // Create a simple summarization chain
  const summarizationPrompt = PromptTemplate.fromTemplate(
    "Summarize the following text in 1-2 sentences:\n\n{text}"
  );

  const summarizationChain = RunnableSequence.from([
    summarizationPrompt,
    chatModel,
  ]);

  // Create a tool from the summarization chain
  const summarizationTool = new DynamicTool({
    name: "text-summarizer",
    description: "Summarizes longer text into 1-2 sentences",
    func: async (text) => {
      const result = await summarizationChain.invoke({ text });
      return result.content;
    },
  });

  // Create an agent with tools
  const tools = [
    new Calculator(),
    summarizationTool,
  ];

  // Pull the prompt
  const prompt = await pull<ChatPromptTemplate>("hwchase17/openai-tools-agent");

# TraceAI Mastra
## Installation

```shell
npm install --save @traceai/mastra
```

## Usage

```typescript
import { Mastra } from "@mastra/core";
import {
  FiOTLPTraceExporter,
  isFiSpan,
} from "@traceai/mastra";

export const mastra = new Mastra({
  // ... other config
  telemetry: {
    serviceName: "traceai-mastra-agent", // you can rename this to whatever you want to appear in the Phoenix UI
    enabled: true,
    export: {
      type: "custom",
      exporter: new FiOTLPTraceExporter({
        url: process.env.PHOENIX_COLLECTOR_ENDPOINT,
        headers: {
          Authorization: `Bearer ${process.env.PHOENIX_API_KEY}`,
        },
        // optional: filter out http, and other node service specific spans
        // they will still be exported to Mastra, but not to the target of
        // this exporter
        spanFilter: isFiSpan,
      }),
    },
  },
});

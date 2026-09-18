const express = require("express");
const cors = require("cors");
const { buildSSEResponse } = require("./sseUtil");
const { serverConfig } = require("./config");
const { createLLMOrchestrationStreamRequest } = require("./streamingService");
const helmet = require("helmet");
const streamQueue = require("./streamQueue");

const app = express();

app.use(cors());
app.use(helmet.hidePoweredBy());
app.use(express.json({ extended: false }));

app.get("/sse/stream/:channelId", (req, res) => {
  const { channelId } = req.params;
  buildSSEResponse({
    req,
    res,
    buildCallbackFunction: ({ connectionId, sender }) => {
      // For streaming SSE, we don't set up an interval
      // Instead, we wait for POST requests to trigger streaming
      console.log(`SSE streaming connection established for channel ${channelId}, connection ${connectionId}`);
      
      // Return cleanup function (no-op for streaming connections)
      return () => {
        console.log(`SSE streaming connection closed for channel ${channelId}, connection ${connectionId}`);
      };
    },
    channelId,
  });
});

app.post("/channels/:channelId/orchestrate/stream", async (req, res) => {
  try {
    const { channelId } = req.params;
    const { message, options = {} } = req.body;

    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "Message string is required" });
    }

    const result = await createLLMOrchestrationStreamRequest({
      channelId,
      message,
      options,
    });

    res.status(200).json(result);
  } catch (error) {
    if (error.message.includes("No active connections found for this channel - request queued")) {
      res.status(202).json({
        message: "Request queued - will be processed when connection becomes available",
        status: "queued",
      });
    } else if (error.message === "No active connections found for this channel") {
      res.status(404).json({ error: error.message });
    } else {
      res.status(500).json({ error: "Failed to start LLM orchestration streaming" });
    }
  }
});

// Cleanup stale stream requests periodically
setInterval(() => {
  const now = Date.now();
  const oneHour = 60 * 60 * 1000;

  for (const [channelId, requests] of streamQueue.queue.entries()) {
    const staleRequests = requests.filter((req) => now - req.timestamp > oneHour || !streamQueue.shouldRetry(req));

    staleRequests.forEach((staleReq) => {
      streamQueue.removeFromQueue(channelId, staleReq.id);
      console.log(`Cleaned up stale stream request for channel ${channelId}`);
    });
  }
}, 5 * 60 * 1000);

const server = app.listen(serverConfig.port, () => {
  console.log(`Notification server running on port ${serverConfig.port}`);
  console.log(`SSE streaming available at: /sse/stream/:channelId`);
  console.log(`LLM orchestration streaming at: /channels/:channelId/orchestrate/stream`);
});

// The SSE GET and the trigger POST are both long-lived by design, so Node's
// default 300s requestTimeout would cut them off mid-answer. Disable the
// per-request cap and let the upstream idle watchdog in streamingService.js
// decide when a stream has genuinely stalled.
server.requestTimeout = 0;
server.headersTimeout = 65_000;
server.keepAliveTimeout = 61_000;

module.exports = server;

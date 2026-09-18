const { v4: uuidv4 } = require('uuid');
const streamQueue = require("./streamQueue");
const { createLLMOrchestrationStreamRequest } = require("./streamingService");
const { activeConnections, abortConnectionRequests } = require("./connectionManager");

// Comment frames are written this often so that every intermediate proxy sees
// traffic and does not close the connection on its idle timer. EventSource
// ignores comment frames, so this is invisible to the browser.
const HEARTBEAT_INTERVAL_MS = Number(
  process.env.SSE_HEARTBEAT_INTERVAL_MS || 15_000
);

function buildSSEResponse({ res, req, buildCallbackFunction, channelId }) {
  addSSEHeader(req, res);
  const heartbeat = keepStreamAlive(res);
  const connectionId = generateConnectionID();
  const sender = buildSender(res);

  activeConnections.set(connectionId, {
    res,
    sender,
    channelId,
    abortControllers: new Set(),
  });

  if (channelId) {
    setTimeout(() => {
      processPendingStreamsForChannel(channelId);
    }, 1000);
  }

  const cleanUp = buildCallbackFunction({ connectionId, sender });

  req.on("close", () => {
    console.log(`Client disconnected from SSE for channel ${channelId}`);
    clearInterval(heartbeat);
    // Cancel any in-flight upstream generation - nobody is left to read it.
    abortConnectionRequests(connectionId);
    activeConnections.delete(connectionId);
    cleanUp?.();
  });
}

function addSSEHeader(req, res) {
  const origin = extractOrigin(req.headers.origin);

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    // Stops nginx-style proxies buffering the stream into a single response.
    'X-Accel-Buffering': 'no',
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Credentials': true,
    'Access-Control-Expose-Headers': 'Origin, X-Requested-With, Content-Type, Cache-Control, Connection, Accept'
  });
}

function extractOrigin(reqOrigin) {
  const corsWhitelist = process.env.CORS_WHITELIST_ORIGINS.split(',');
  const whitelisted = corsWhitelist.indexOf(reqOrigin) !== -1;
  return whitelisted ? reqOrigin : '*';
}

/**
 * Keep the SSE connection warm with periodic comment frames.
 *
 * Previously a single `res.write('')`, which did nothing beyond the initial
 * flush - any proxy between the browser and this server would still time the
 * connection out during a long generation pause.
 *
 * @returns {NodeJS.Timeout} interval handle; the caller must clear it on close.
 */
function keepStreamAlive(res) {
  res.write('');
  const heartbeat = setInterval(() => {
    try {
      // A `:` line is an SSE comment: ignored by EventSource, but it is traffic.
      res.write(': ping\n\n');
      if (typeof res.flush === "function") {
        res.flush();
      }
    } catch (error) {
      console.error("SSE heartbeat write failed:", error);
      clearInterval(heartbeat);
    }
  }, HEARTBEAT_INTERVAL_MS);

  // Do not hold the event loop open purely for a heartbeat.
  heartbeat.unref?.();
  return heartbeat;
}

function generateConnectionID() {
  const connectionId = uuidv4();
  console.log(`New client connected with connectionId: ${connectionId}`);
  return connectionId;
}

function buildSender(res) {
  return (data) => {
    try {
      const formattedData = typeof data === "string" ? data : JSON.stringify(data);
      res.write(`data: ${formattedData}\n\n`);
      if (typeof res.flush === "function") {
        res.flush();
      }
    } catch (error) {
      console.error("SSE write error:", error);
    }
  };
}

function processPendingStreamsForChannel(channelId) {
  const pendingRequests = streamQueue.getPendingRequests(channelId);

  if (pendingRequests.length > 0) {
    pendingRequests.forEach(async (requestData) => {
      if (streamQueue.shouldRetry(requestData)) {
        try {
          
          await createLLMOrchestrationStreamRequest({
            channelId,
            message: requestData.message,
            options: requestData.options,
          });

          streamQueue.removeFromQueue(channelId, requestData.id);
        } catch (error) {
          console.error(`Failed to process queued stream for channel ${channelId}:`, error);
          streamQueue.incrementRetryCount(channelId, requestData.id);
        }
      } else {
        streamQueue.removeFromQueue(channelId, requestData.id);
      }
    });
  }
}

module.exports = {
  activeConnections,
  buildSSEResponse,
  processPendingStreamsForChannel,
};

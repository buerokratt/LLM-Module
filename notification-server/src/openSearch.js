const { activeConnections } = require("./connectionManager");
const streamQueue = require("./streamQueue");

/**
 * Stream LLM orchestration response to connected clients
 * @param {Object} params - Request parameters
 * @param {string} params.channelId - Channel identifier
 * @param {string} params.message - User message
 * @param {Object} params.options - Additional options (authorId, conversationHistory, url)
 */
async function createLLMOrchestrationStreamRequest({ channelId, message, options = {} }) {
  const connections = Array.from(activeConnections.entries()).filter(
    ([_, connData]) => connData.channelId === channelId
  );

  console.log(`Active connections for channel ${channelId}:`, connections.length);

  if (connections.length === 0) {
    const requestId = streamQueue.addToQueue(channelId, { message, options });
    
    if (streamQueue.shouldRetry({ retryCount: 0 })) {
      throw new Error("No active connections found for this channel - request queued");
    } else {
      throw new Error("No active connections found for this channel");
    }
  }

  console.log(`Streaming LLM orchestration for channel ${channelId} to ${connections.length} connections`);

  try {
    const responsePromises = connections.map(async ([connectionId, connData]) => {
      const { sender } = connData;

      try {
        // Construct OrchestrationRequest payload
        const orchestrationPayload = {
          chatId: channelId,
          message: message,
          authorId: options.authorId || `user-${channelId}`,
          conversationHistory: options.conversationHistory || [],
          url: options.url || "sse-stream-context",
          environment: options.environment || "production",
          connection_id: options.connection_id
        };

        console.log(`Calling LLM orchestration stream for channel ${channelId}`);

        // Call the LLM orchestration streaming endpoint
        const response = await fetch(`${process.env.LLM_ORCHESTRATOR_URL || 'http://llm-orchestration-service:8100'}/orchestrate/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(orchestrationPayload),
        });

        if (!response.ok) {
          throw new Error(`LLM Orchestration API error: ${response.status} ${response.statusText}`);
        }

        if (!activeConnections.has(connectionId)) {
          return;
        }

        // Send stream start notification
        sender({
          type: "stream_start",
          streamId: channelId,
          channelId,
          isComplete:false
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          if (!activeConnections.has(connectionId)) break;

          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep the incomplete line in buffer

          for (const line of lines) {
            if (!line.trim()) continue;
            if (!line.startsWith('data: ')) continue;

            try {
              const data = JSON.parse(line.slice(6)); // Remove 'data: ' prefix
              const content = data.payload?.content;
              
              if (!content) continue;

              if (content === "END") {
                // Stream completed
                sender({
                  type: "stream_end",
                  streamId: channelId,
                  channelId,
                  isComplete:true
                });
                break;
              }

              // Regular token - send to client
              sender({
                type: "stream_chunk",
                content: content,
                streamId: channelId,
                channelId,
                isComplete:false
              });

            } catch (parseError) {
              console.error(`Failed to parse SSE data for channel ${channelId}:`, parseError, line);
            }
          }
        }

      } catch (error) {
        console.error(`Streaming error for connection ${connectionId}:`, error);
        if (activeConnections.has(connectionId)) {
          sender({
            type: "stream_error",
            error: error.message,
            streamId: channelId,
            channelId,
            isComplete:true
          });
        }
      }
    });

    await Promise.all(responsePromises);
    return { success: true, message: "Stream completed" };

  } catch (error) {
    console.error(`Error in createLLMOrchestrationStreamRequest:`, error);
    throw error;
  }
}

module.exports = {
  createLLMOrchestrationStreamRequest,
};

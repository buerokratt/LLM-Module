import { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import { ChoiceButton } from 'services/inference';

const getNotificationNodeUrl = (): string => {
  const value = import.meta.env.REACT_APP_NOTIFICATION_NODE_URL;
  if (!value) {
    throw new Error(
      'Environment variable REACT_APP_NOTIFICATION_NODE_URL is not defined. ' +
        'Please set it to the base URL of the notification service to enable streaming responses.'
    );
  }
  return value;
};
const notificationNodeUrl = getNotificationNodeUrl();
console.log(notificationNodeUrl);

// The trigger POST is held open by the notification server for the full
// generation, so it needs a ceiling well above any realistic answer time.
const TRIGGER_POST_TIMEOUT_MS = 600_000;

// How long to keep waiting on SSE after the trigger POST has failed. A failed
// POST is not proof that generation failed - it may already be streaming - but
// if nothing has arrived by now, nothing is coming: the POST failed before the
// server ever dispatched to the relay, so no stream_end or stream_error will
// ever be sent and without this the UI would wait forever.
const TRIGGER_FAILURE_GRACE_MS = 15_000;

// Messages that prove the backend actually engaged this stream. Heartbeats are
// SSE comment frames, which EventSource never surfaces, so they cannot count.
const STREAM_EVENT_TYPES = new Set([
  'stream_start',
  'stream_chunk',
  'stream_end',
  'stream_error',
]);

// Typewriter pacing.
//
// Output guardrails validate the answer in blocks before releasing it, so tokens
// reach the browser in bursts (roughly 200, then 150, then the tail) rather than
// one at a time. Rendering each burst the instant it lands makes the answer snap
// onto the screen. Instead we queue arriving tokens and drain them on a timer,
// which gives a steady word-by-word effect without weakening the guardrails or
// paying for the extra validation calls that smaller server-side chunks cost.
//
// Typing speed. This is the knob to turn if the effect feels too fast or slow -
// higher is faster. Around 25/s reads like brisk typing; 40+/s starts to look
// like the text is simply appearing.
const TYPING_TOKENS_PER_SECOND = 25;
// Ceiling on how far rendering may fall behind the stream. If a burst arrives
// faster than the typing speed, the drain rate rises so the backlog still clears
// within this budget rather than typing on long after the answer is complete.
// Approximate: setInterval fires late under load, so expect ~15-20% over.
const MAX_CATCH_UP_SECONDS = 8;

const DRAIN_INTERVAL_MS = Math.round(1000 / TYPING_TOKENS_PER_SECOND);
const DRAIN_TARGET_TICKS = Math.round(
  (MAX_CATCH_UP_SECONDS * 1000) / DRAIN_INTERVAL_MS
);

interface StreamingOptions {
  authorId: string;
  conversationHistory: Array<{ authorRole: string; message: string; timestamp: string }>;
  url: string;
}

interface UseStreamingResponseReturn {
  startStreaming: (message: string, options: StreamingOptions, onToken: (token: string) => void, onComplete: () => void, onError: (error: string) => void, onButtons?: (buttons: ChoiceButton[]) => void) => Promise<void>;
  stopStreaming: () => void;
  isStreaming: boolean;
}

export const useStreamingResponse = (channelId: string): UseStreamingResponseReturn => {
  const [isStreaming, setIsStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Typewriter state
  const queueRef = useRef<string[]>([]);
  const drainTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Tokens emitted per tick. Only ever raised during a run: recomputing it from
  // the shrinking queue each tick would decay the rate geometrically and stretch
  // a large backlog out well beyond MAX_CATCH_UP_SECONDS.
  const drainRateRef = useRef(1);
  const streamEndedRef = useRef(false);
  const pendingButtonsRef = useRef<ChoiceButton[] | null>(null);
  // Set by any stream event; read by the trigger-POST fallback below.
  const sawStreamEventRef = useRef(false);
  const triggerFallbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onTokenRef = useRef<(token: string) => void>(() => {});
  const onCompleteRef = useRef<() => void>(() => {});
  const onButtonsRef = useRef<((buttons: ChoiceButton[]) => void) | undefined>(undefined);

  const stopDrain = useCallback(() => {
    if (drainTimerRef.current) {
      clearInterval(drainTimerRef.current);
      drainTimerRef.current = null;
    }
  }, []);

  const clearTriggerFallback = useCallback(() => {
    if (triggerFallbackTimerRef.current) {
      clearTimeout(triggerFallbackTimerRef.current);
      triggerFallbackTimerRef.current = null;
    }
  }, []);

  // Drop anything not yet rendered. Used when a guardrail blocks the answer or
  // the user cancels: text the rail rejected must never reach the screen.
  const discardQueue = useCallback(() => {
    queueRef.current = [];
    pendingButtonsRef.current = null;
    drainRateRef.current = 1;
    stopDrain();
  }, [stopDrain]);

  const startDrain = useCallback(() => {
    if (drainTimerRef.current) return;

    drainTimerRef.current = setInterval(() => {
      const queue = queueRef.current;

      if (queue.length === 0) {
        if (streamEndedRef.current) {
          stopDrain();
          if (pendingButtonsRef.current?.length && onButtonsRef.current) {
            onButtonsRef.current(pendingButtonsRef.current);
            pendingButtonsRef.current = null;
          }
          setIsStreaming(false);
          onCompleteRef.current();
        }
        return;
      }

      drainRateRef.current = Math.max(
        drainRateRef.current,
        Math.ceil(queue.length / DRAIN_TARGET_TICKS)
      );
      onTokenRef.current(queue.splice(0, drainRateRef.current).join(''));
    }, DRAIN_INTERVAL_MS);
  }, [stopDrain]);

  const stopStreaming = useCallback(() => {
    if (eventSourceRef.current) {
      console.log('[SSE] Closing connection');
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    clearTriggerFallback();
    discardQueue();
    setIsStreaming(false);
  }, [discardQueue, clearTriggerFallback]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (drainTimerRef.current) {
        clearInterval(drainTimerRef.current);
      }
      if (triggerFallbackTimerRef.current) {
        clearTimeout(triggerFallbackTimerRef.current);
      }
    };
  }, []);

  const startStreaming = useCallback(
    async (
      message: string,
      options: StreamingOptions,
      onToken: (token: string) => void,
      onComplete: () => void,
      onError: (error: string) => void,
      onButtons?: (buttons: ChoiceButton[]) => void
    ) => {
      console.log('[SSE] Starting streaming for channel:', channelId);

      // Close any existing connection
      stopStreaming();

      // Reset typewriter state for this run
      queueRef.current = [];
      drainRateRef.current = 1;
      streamEndedRef.current = false;
      pendingButtonsRef.current = null;
      sawStreamEventRef.current = false;
      onTokenRef.current = onToken;
      onCompleteRef.current = onComplete;
      onButtonsRef.current = onButtons;

      try {
        // Step 1: Open SSE connection FIRST
        const sseUrl = `${notificationNodeUrl}/sse/stream/${channelId}`;
        console.log('[SSE] Connecting to:', sseUrl);

        const eventSource = new EventSource(sseUrl);
        eventSourceRef.current = eventSource;

        eventSource.onopen = () => {
          console.log('[SSE] Connection opened');
        };

        eventSource.onmessage = (event) => {
          console.log('[SSE] Message received:', event.data);

          try {
            const data = JSON.parse(event.data);

            if (STREAM_EVENT_TYPES.has(data.type)) {
              // The backend is talking to us, so the trigger-POST fallback must
              // not fire - whatever happens next arrives over this connection.
              sawStreamEventRef.current = true;
              clearTriggerFallback();
            }

            if (data.type === 'stream_start') {
              console.log('[SSE] Stream started');
              setIsStreaming(true);
              startDrain();
            } else if (data.type === 'stream_chunk' && data.content) {
              // Queue rather than render, so bursts play out word by word.
              queueRef.current.push(data.content);
              if (data.buttons && data.buttons.length > 0) {
                // Held back until the text finishes typing, so choices do not
                // appear above a half-rendered answer.
                pendingButtonsRef.current = data.buttons;
              }
              startDrain();
            } else if (data.type === 'stream_end') {
              console.log('[SSE] Stream ended');
              eventSource.close();
              eventSourceRef.current = null;
              // Do not fire onComplete yet - let the queue finish rendering.
              streamEndedRef.current = true;
              startDrain();
            } else if (data.type === 'stream_error') {
              console.error('[SSE] Stream error:', data.error);
              // Discard unrendered text: a guardrail may have just rejected it.
              discardQueue();
              setIsStreaming(false);
              eventSource.close();
              eventSourceRef.current = null;
              onError(data.error || 'Stream error occurred');
            }
          } catch (e) {
            console.error('[SSE] Failed to parse message:', e);
          }
        };

        eventSource.onerror = (err) => {
          console.error('[SSE] Connection error:', err);
          // Otherwise a pending fallback would later report a second error for
          // the same failed run.
          clearTriggerFallback();
          discardQueue();
          setIsStreaming(false);
          eventSource.close();
          eventSourceRef.current = null;
          onError('Connection error');
        };

        // Step 2: Wait a moment for SSE connection to establish, then trigger the stream
        await new Promise(resolve => setTimeout(resolve, 500));

        // Step 3: POST to trigger streaming.
        // Note: this request stays open for the whole generation, so it can fail
        // (gateway 504, network blip) while the SSE stream is perfectly healthy.
        const postUrl = `${notificationNodeUrl}/channels/${channelId}/orchestrate/stream`;
        console.log('[API] Triggering stream:', postUrl);

        try {
          await axios.post(
            postUrl,
            { message, options },
            { timeout: TRIGGER_POST_TIMEOUT_MS }
          );
          console.log('[API] Stream triggered successfully');
        } catch (postErr) {
          // Do NOT tear down the EventSource here. The answer arrives over SSE,
          // not in this response body; killing the stream on a POST failure is
          // what turned a slow answer into a truncated one. Let stream_end /
          // stream_error decide, and only surface an error if neither arrives.
          console.warn(
            '[API] Trigger POST failed; keeping SSE open and waiting for stream events:',
            postErr
          );

          const postErrMessage =
            postErr instanceof Error ? postErr.message : 'Failed to start streaming';

          if (!eventSourceRef.current) {
            // SSE already gone - nothing left to wait for.
            onError(postErrMessage);
            return;
          }

          // Bound the wait. If the POST failed before the server dispatched to
          // the relay (a 4xx, or no active connection so the request was only
          // queued), no stream event will ever arrive and there is nothing to
          // end the stream - so give up once the grace period expires.
          clearTriggerFallback();
          triggerFallbackTimerRef.current = setTimeout(() => {
            triggerFallbackTimerRef.current = null;
            if (sawStreamEventRef.current) return;

            console.error(
              `[SSE] No stream events ${TRIGGER_FAILURE_GRACE_MS}ms after trigger POST failed; giving up`
            );
            discardQueue();
            setIsStreaming(false);
            if (eventSourceRef.current) {
              eventSourceRef.current.close();
              eventSourceRef.current = null;
            }
            onError(postErrMessage);
          }, TRIGGER_FAILURE_GRACE_MS);
        }

      } catch (err) {
        console.error('[SSE] Error starting stream:', err);
        stopStreaming();
        onError(err instanceof Error ? err.message : 'Failed to start streaming');
      }
    },
    [channelId, stopStreaming, startDrain, discardQueue, clearTriggerFallback]
  );

  return {
    startStreaming,
    stopStreaming,
    isStreaming,
  };
};

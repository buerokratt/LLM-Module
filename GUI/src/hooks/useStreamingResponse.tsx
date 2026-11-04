import { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useNotificationEvents, useNotificationsClient } from '@buerokratt-ria/notifications/react';
import type { NotificationData } from '@buerokratt-ria/notifications';
import apiDev from 'services/api-dev';
import { ChoiceButton } from 'services/inference';

interface StreamingOptions {
  readonly authorId: string;
  readonly conversationHistory: Array<{ authorRole: string; message: string; timestamp: string }>;
  readonly url: string;
  readonly environment?: string;
  readonly connection_id?: string;
}

interface StreamPayload {
  readonly content?: string;
  readonly buttons?: ChoiceButton[];
  readonly error?: string;
}

interface StartStreamRequest {
  readonly chatUuid: string;
  readonly message: string;
  readonly conversationHistory: StreamingOptions['conversationHistory'];
  readonly environment?: string;
  readonly connectionId?: string;
}

interface StreamRun {
  requestId?: string;
  cancelled: boolean;
  finished: boolean;
  cancelSent: boolean;
}

interface StartStreamResponse {
  readonly requestId: string;
}

const cancelBackendStream = (run: StreamRun) => {
  if (!run.requestId || run.finished || run.cancelSent) return;
  run.cancelSent = true;
  void apiDev.post('/inference/stream-cancel', { requestId: run.requestId }).catch(() => {
    console.warn('Failed to cancel the backend stream');
  });
};

const STREAM_EVENTS = ['stream_start', 'stream_chunk', 'stream_complete', 'stream_error'] as const;
const TYPING_TOKENS_PER_SECOND = 25;
const MAX_CATCH_UP_SECONDS = 8;

const DRAIN_INTERVAL_MS = Math.round(1000 / TYPING_TOKENS_PER_SECOND);
const DRAIN_TARGET_TICKS = Math.round(
  (MAX_CATCH_UP_SECONDS * 1000) / DRAIN_INTERVAL_MS
);

export const useStreamingResponse = (chatUuid: string) => {
  const client = useNotificationsClient();
  const { connect, getState, subscribeToState } = client;
  const { mutateAsync: startStreamRequest, reset: resetStreamRequest } = useMutation({
    mutationFn: ({ chatUuid, message, conversationHistory, environment, connectionId }: StartStreamRequest) =>
      apiDev.post<StartStreamResponse | { response: StartStreamResponse }>('/inference/stream', {
        chatUuid,
        message,
        conversationHistory,
        environment,
        connectionId,
      }),
    retry: false,
  });
  const [isStreaming, setIsStreaming] = useState(false);
  const active = useRef(false);
  const ended = useRef(false);
  const queue = useRef<string[]>([]);
  const buttons = useRef<ChoiceButton[]>([]);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const rate = useRef(1);
  const requestAbort = useRef<AbortController | null>(null);
  const currentRun = useRef<StreamRun | null>(null);
  const callbacks = useRef<{
    onToken: (token: string) => void;
    onComplete: () => void;
    onError: (error: string) => void;
    onButtons?: (buttons: ChoiceButton[]) => void;
  }>();

  const stopStreaming = useCallback(() => {
    const run = currentRun.current;
    currentRun.current = null;
    if (run && !run.finished) {
      run.cancelled = true;
      cancelBackendStream(run);
    }
    active.current = false;
    requestAbort.current?.abort();
    requestAbort.current = null;
    resetStreamRequest();
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
    queue.current = [];
    buttons.current = [];
    ended.current = false;
    rate.current = 1;
    setIsStreaming(false);
  }, [resetStreamRequest]);

  useEffect(() => {
    connect({ chatUuids: [chatUuid] });
    return () => {
      stopStreaming();
      connect();
    };
  }, [chatUuid, connect, stopStreaming]);

  const startDrain = useCallback(() => {
    if (timer.current) return;
    timer.current = setInterval(() => {
      if (queue.current.length) {
        rate.current = Math.max(rate.current, Math.ceil(queue.current.length / DRAIN_TARGET_TICKS));
        callbacks.current?.onToken(queue.current.splice(0, rate.current).join(''));
      } else if (ended.current) {
        const pendingButtons = buttons.current;
        const currentCallbacks = callbacks.current;
        stopStreaming();
        if (pendingButtons.length) currentCallbacks?.onButtons?.(pendingButtons);
        currentCallbacks?.onComplete();
      }
    }, DRAIN_INTERVAL_MS);
  }, [stopStreaming]);

  useNotificationEvents<NotificationData<StreamPayload>>({
    eventTypes: STREAM_EVENTS,
    listener: ({ type, data }) => {
      if (!active.current || data.recipient !== 'CHAT' || data.recipientUuid !== chatUuid) return;
      const payload = data.payload;
      if (type === 'stream_chunk') {
        if (payload.content) queue.current.push(payload.content);
        if (payload.buttons?.length) buttons.current = payload.buttons;
        startDrain();
      } else if (type === 'stream_complete') {
        if (currentRun.current) currentRun.current.finished = true;
        ended.current = true;
        startDrain();
      } else if (type === 'stream_error') {
        if (currentRun.current) currentRun.current.finished = true;
        const onError = callbacks.current?.onError;
        stopStreaming();
        onError?.(payload.error || 'Stream error occurred');
      }
    },
  });

  const startStreaming = useCallback(async (
    message: string,
    options: StreamingOptions,
    onToken: (token: string) => void,
    onComplete: () => void,
    onError: (error: string) => void,
    onButtons?: (buttons: ChoiceButton[]) => void,
  ) => {
    stopStreaming();
    const run: StreamRun = { cancelled: false, finished: false, cancelSent: false };
    currentRun.current = run;
    const abort = new AbortController();
    requestAbort.current = abort;
    callbacks.current = { onToken, onComplete, onError, onButtons };
    active.current = true;
    setIsStreaming(true);
    try {
      connect({ chatUuids: [chatUuid] });
      // The node drops chat events until the chat subscription is connected.
      if (getState().status !== 'connected') {
        await new Promise<void>((resolve, reject) => {
          let unsubscribe = () => { };
          const cleanup = () => { clearTimeout(timeout); unsubscribe(); abort.signal.removeEventListener('abort', cancel); };
          const cancel = () => { cleanup(); reject(new Error('Stream cancelled')); };
          const timeout = setTimeout(() => { cleanup(); reject(new Error('Notification connection timed out')); }, 10000);
          unsubscribe = subscribeToState(() => {
            const state = getState();
            if (state.status === 'connected') { cleanup(); resolve(); }
            else if (state.status === 'session-expired') { cleanup(); reject(new Error('Session expired')); }
          });
          abort.signal.addEventListener('abort', cancel, { once: true });
          if (abort.signal.aborted) cancel();
          else if (getState().status === 'connected') { cleanup(); resolve(); }
        });
      }
      // Keep the trigger response available even if Stop is pressed before 202:
      // its requestId is needed to cancel the accepted backend task.
      const response = await startStreamRequest({
        chatUuid,
        message,
        conversationHistory: options.conversationHistory,
        environment: options.environment,
        connectionId: options.connection_id,
      });
      const accepted = 'response' in response.data ? response.data.response : response.data;
      run.requestId = accepted.requestId;
      if (run.cancelled) cancelBackendStream(run);
    } catch (error) {
      if (abort.signal.aborted) return;
      stopStreaming();
      onError(error instanceof Error ? error.message : 'Failed to start streaming');
    }
  }, [chatUuid, connect, getState, startStreamRequest, subscribeToState, stopStreaming]);

  return { startStreaming, stopStreaming, isStreaming };
};

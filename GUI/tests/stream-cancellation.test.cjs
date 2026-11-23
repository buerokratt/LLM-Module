const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

// Exercise the hook's async requests and event callbacks without a browser or node.
function mountHook({ pending = false } = {}) {
  const posts = [];
  const cleanups = [];
  const intervals = new Map();
  let listener;
  let resolveStart;
  const accepted = pending
    ? new Promise((resolve) => { resolveStart = resolve; })
    : Promise.resolve({ data: { response: { requestId: 'request-1' } } });
  const api = {
    post: (url, body) => {
      posts.push({ url, body });
      return url === '/inference/stream' ? accepted : Promise.resolve({ data: true });
    },
  };
  const client = {
    connect() {},
    getState: () => ({ status: 'connected' }),
    subscribeToState: () => () => {},
  };
  const modules = {
    react: {
      useCallback: (fn) => fn,
      useEffect: (fn) => { cleanups.push(fn()); },
      useRef: (value) => ({ current: value }),
      useState: (value) => [value, () => {}],
    },
    '@tanstack/react-query': {
      useMutation: ({ mutationFn }) => ({ mutateAsync: mutationFn, reset() {} }),
    },
    '@buerokratt-ria/notifications/react': {
      useNotificationsClient: () => client,
      useNotificationEvents: (options) => { listener = options.listener; },
    },
    'services/api-dev': { default: api },
  };
  const context = {
    exports: {},
    require: (name) => {
      assert.ok(name in modules, `Unexpected import: ${name}`);
      return modules[name];
    },
    AbortController,
    console,
    setTimeout,
    clearTimeout,
    setInterval: (fn) => { const id = intervals.size + 1; intervals.set(id, fn); return id; },
    clearInterval: (id) => intervals.delete(id),
  };
  const source = readFileSync(path.join(__dirname, '../src/hooks/useStreamingResponse.tsx'), 'utf8');
  const output = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  });
  vm.runInNewContext(output.outputText, context);
  const hook = context.exports.useStreamingResponse('chat-1');
  return {
    hook,
    posts,
    start: () => hook.startStreaming('tere', { conversationHistory: [] }, () => {}, () => {}, () => {}),
    accept: () => resolveStart({ data: { response: { requestId: 'request-1' } } }),
    event: (type) => listener({ type, data: { recipient: 'CHAT', recipientUuid: 'chat-1', payload: {} } }),
    drain: () => { for (const fn of [...intervals.values()]) fn(); },
    unmount: () => { for (const cleanup of cleanups) cleanup?.(); },
    cancellations: () => posts.filter(({ url }) => url === '/inference/stream-cancel'),
  };
}

test('Stop cancels the accepted backend request once', async () => {
  const view = mountHook();
  await view.start();
  view.hook.stopStreaming();
  view.hook.stopStreaming();
  assert.equal(view.cancellations().length, 1);
  assert.equal(view.cancellations()[0].body.requestId, 'request-1');
});

test('Stop before 202 cancels as soon as requestId arrives', async () => {
  const view = mountHook({ pending: true });
  const started = view.start();
  view.hook.stopStreaming();
  assert.equal(view.cancellations().length, 0);
  view.accept();
  await started;
  assert.equal(view.cancellations().length, 1);
  assert.equal(view.cancellations()[0].body.requestId, 'request-1');
});

test('Normal completion does not send cancellation', async () => {
  const view = mountHook();
  await view.start();
  view.event('stream_complete');
  view.drain();
  view.hook.stopStreaming();
  assert.equal(view.cancellations().length, 0);
});

test('A terminal error does not send cancellation', async () => {
  const view = mountHook();
  await view.start();
  view.event('stream_error');
  assert.equal(view.cancellations().length, 0);
});

test('Leaving the page cancels an active backend request', async () => {
  const view = mountHook();
  await view.start();
  view.unmount();
  assert.equal(view.cancellations().length, 1);
});

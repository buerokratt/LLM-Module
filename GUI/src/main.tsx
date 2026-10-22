import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import {
  QueryClient,
  QueryClientProvider,
  QueryFunction,
} from '@tanstack/react-query';

import App from './App';
import api from 'services/api';
import apiDev from 'services/api-dev';
import apigeneric from 'services/apigeneric';
import auth from 'services/auth';
import { ToastProvider } from 'context/ToastContext';
import 'styles/main.scss';
import '../i18n';
import { CookiesProvider } from 'react-cookie';
import { DialogProvider } from 'context/DialogContext';

const isLocal = import.meta.env.REACT_APP_LOCAL?.toLowerCase() === 'true';

const defaultQueryFn: QueryFunction | undefined = async ({ queryKey }) => {
  if (isLocal && queryKey.includes('prod')) {
    const { data } = await apigeneric.get(queryKey[0] as string);
    return data.response;
  }
  if (queryKey.includes('prod')) {
    const { data } = await apiDev.get(queryKey[0] as string);
    return data;
  }
  if (queryKey[1] === 'auth') {
    const { data } = await auth.get(queryKey[0] as string);
    return data;
  }

  const { data } = await api.get(queryKey[0] as string);
  return data;
};

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: defaultQueryFn,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename={import.meta.env.BASE_URL}>
        <DialogProvider>
          <ToastProvider>
            <CookiesProvider>
              <App />
            </CookiesProvider>
          </ToastProvider>
        </DialogProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);

import apiDev from './api-dev';

export interface PlatformOption {
  id: number;
  value: string;
  label: string;
  is_active: boolean;
  created_at: string;
}

export interface ModelOption {
  id: number;
  value: string;
  label: string;
  platform_id: number;
  platform_key: string;
  platform_name: string;
  is_active: boolean;
  created_at: string;
}

// Get all LLM platforms
export async function getLLMPlatforms(): Promise<PlatformOption[]> {
  const { data } = await apiDev.get('/rag-search/llm/platforms');
  return data?.response;
}

// Get LLM models by platform
export async function getLLMModels(platformKey?: string): Promise<ModelOption[]> {
  const { data } = await apiDev.get('/rag-search/llm/models', {
    params: platformKey ? { platform_key: platformKey } : {}
  });
  return data?.response;
}

// Get all embedding platforms
export async function getEmbeddingPlatforms(): Promise<PlatformOption[]> {
  const { data } = await apiDev.get('/rag-search/embedding/platforms');
  return data?.response;
}

// Get embedding models by platform
export async function getEmbeddingModels(platformKey?: string): Promise<ModelOption[]> {
  const { data } = await apiDev.get('/rag-search/embedding/models', {
    params: platformKey ? { embedding_platform_key: platformKey } : {}
  });
  return data?.response;
}

// Get all LLM models
export async function getAllLLMModels(): Promise<ModelOption[]> {
  const { data } = await apiDev.get('/rag-search/llm/models-list');
  return data?.response;
}
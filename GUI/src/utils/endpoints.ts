export const userManagementEndpoints = {
  FETCH_USERS: (): string => `/rag-search/accounts/users`,
  ADD_USER: (): string => `/rag-search/accounts/add`,
  CHECK_ACCOUNT_AVAILABILITY: (): string => `/rag-search/accounts/exists`,
  EDIT_USER: (): string => `/rag-search/accounts/edit`,
  DELETE_USER: (): string => `/rag-search/accounts/delete`,
  FETCH_USER_ROLES: (): string => `/rag-search/accounts/user-role`,
};


export const authEndpoints = {
  GET_EXTENDED_COOKIE: () :string => `/rag-search/auth/jwt/extend`,
  LOGOUT: (): string => `/rag-search/accounts/logout`
}

export const llmConnectionsEndpoints = {
  FETCH_LLM_CONNECTIONS_PAGINATED: (): string => `/rag-search/llm-connections/list`,
  FETCH_ALL_LLM_CONNECTIONS_PAGINATED: (): string => `/rag-search/llm-connections/all`,
  GET_LLM_CONNECTION: (): string => `/rag-search/llm-connections/get`,
  GET_PRODUCTION_CONNECTION: (): string => `/rag-search/llm-connections/production`,
  CREATE_LLM_CONNECTION: (): string => `/rag-search/llm-connections/add`,
  UPDATE_LLM_CONNECTION: (): string => `/rag-search/llm-connections/edit`,
  UPDATE_LLM_CONNECTION_STATUS: (): string => `/rag-search/llm-connections/update-status`,
  DELETE_LLM_CONNECTION: (): string => `/rag-search/llm-connections/delete`,
  CHECK_BUDGET_STATUS: (): string => `/rag-search/llm-connections/cost/check`,
}

export const inferenceEndpoints = {
  VIEW_TEST_INFERENCE_RESULT: (): string => `/rag-search/inference/test`,
  // Remove after testing
  PRODUCTION_INFERENCE: (): string => `/rag-search/inference/production`,
}

export const vaultEndpoints = {
  CREATE_VAULT_SECRET: (): string => `/rag-search/vault/secret/create`,
  DELETE_VAULT_SECRET: (): string => `/rag-search/vault/secret/delete`,
}

export const promptConfigurationEndpoints = {
  GET_PROMPT_CONFIGURATION: (): string => `/rag-search/prompt-configuration/get`,
  SAVE_PROMPT_CONFIGURATION: (): string => `/rag-search/prompt-configuration/save`,
}

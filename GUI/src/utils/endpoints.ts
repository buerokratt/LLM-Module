export const userManagementEndpoints = {
  FETCH_USERS: (): string => `/accounts/users`,
  ADD_USER: (): string => `/accounts/add`,
  CHECK_ACCOUNT_AVAILABILITY: (): string => `/accounts/exists`,
  EDIT_USER: (): string => `/accounts/edit`,
  DELETE_USER: (): string => `/accounts/delete`,
  FETCH_USER_ROLES: (): string => `/accounts/user-role`,
};


export const authEndpoints = {
  GET_EXTENDED_COOKIE: () :string => `/auth/jwt/extend`,
  LOGOUT: (): string => `/accounts/logout`
}

export const llmConnectionsEndpoints = {
  FETCH_LLM_CONNECTIONS_PAGINATED: (): string => `/llm-connections/list`,
  FETCH_ALL_LLM_CONNECTIONS_PAGINATED: (): string => `/llm-connections/all`,
  GET_LLM_CONNECTION: (): string => `/llm-connections/get`,
  GET_PRODUCTION_CONNECTION: (): string => `/llm-connections/production`,
  CREATE_LLM_CONNECTION: (): string => `/llm-connections/add`,
  UPDATE_LLM_CONNECTION: (): string => `/llm-connections/edit`,
  UPDATE_LLM_CONNECTION_STATUS: (): string => `/llm-connections/update-status`,
  DELETE_LLM_CONNECTION: (): string => `/llm-connections/delete`,
  CHECK_BUDGET_STATUS: (): string => `/llm-connections/cost/check`,
}

export const inferenceEndpoints = {
  VIEW_TEST_INFERENCE_RESULT: (): string => `/inference/test`,
  // Remove after testing
  PRODUCTION_INFERENCE: (): string => `/inference/production`,
}

export const vaultEndpoints = {
  CREATE_VAULT_SECRET: (): string => `/vault/secret/create`,
  DELETE_VAULT_SECRET: (): string => `/vault/secret/delete`,
}

export const promptConfigurationEndpoints = {
  GET_PROMPT_CONFIGURATION: (): string => `/prompt-configuration/get`,
  SAVE_PROMPT_CONFIGURATION: (): string => `/prompt-configuration/save`,
}

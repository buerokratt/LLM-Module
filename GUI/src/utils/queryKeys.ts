import { PaginationState, SortingState } from '@tanstack/react-table';
import { LLMConnectionFilters, LegacyLLMConnectionFilters, ProductionConnectionFilters } from 'services/llmConnections';
import { InferenceRequest } from 'services/inference';


export const authQueryKeys = {
  USER_DETAILS: () => ['auth/jwt/userinfo', 'prod'],
  USER_ROLES: (): string[] => ['/accounts/user-role', 'prod'],

};


export const userManagementQueryKeys = {
  getAllEmployees: function (
    pagination?: PaginationState,
    sorting?: SortingState
  ) {
    return ['accounts/users', pagination, sorting].filter(
      (val) => val !== undefined
    );
  },
};

export const llmConnectionsQueryKeys = {
  all: () => ['llm-connections'] as const,
  lists: () => [...llmConnectionsQueryKeys.all(), 'list'] as const,
  list: (filters: LegacyLLMConnectionFilters) => [...llmConnectionsQueryKeys.lists(), filters] as const,
  paginatedLists: () => [...llmConnectionsQueryKeys.all(), 'paginated-list'] as const,
  paginatedList: (filters: LLMConnectionFilters) => [...llmConnectionsQueryKeys.paginatedLists(), filters] as const,
  details: () => [...llmConnectionsQueryKeys.all(), 'detail'] as const,
  detail: (id: string | number) => [...llmConnectionsQueryKeys.details(), id] as const,
  budgetStatus: () => [...llmConnectionsQueryKeys.all(), 'budget-status'] as const,
  production: (filters?: ProductionConnectionFilters) => [...llmConnectionsQueryKeys.all(), 'production', filters] as const,
};

export const inferenceQueryKeys = {
  all: () => ['inference'] as const,
  results: () => [...inferenceQueryKeys.all(), 'results'] as const,
  result: (request: InferenceRequest) => [...inferenceQueryKeys.results(), request] as const,
};

export const promptConfigurationQueryKeys = {
  all: () => ['prompt-configuration'] as const,
  current: () => [...promptConfigurationQueryKeys.all(), 'current'] as const,
};

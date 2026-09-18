import apiDev from './api-dev';
import { llmConnectionsEndpoints, vaultEndpoints } from 'utils/endpoints';
import { removeCommasFromNumber } from 'utils/commonUtils';
import { maskSensitiveKey } from 'utils/llmConnectionsUtils';
import { encryptLLMCredentials } from 'utils/encryption';

export interface LLMConnection {
  id: number;
  vaultUuid?: string;
  connectionName: string;
  llmPlatform: string;
  llmModel: string;
  embeddingPlatform: string;
  embeddingModel: string;
  monthlyBudget: number;
  warnBudgetThreshold: number;
  stopBudgetThreshold: number;
  disconnectOnBudgetExceed: boolean;
  environment: string;
  connectionStatus: 'active' | 'inactive';
  createdAt: string;
  updatedAt: string;
  totalPages?: number;
  budgetStatus: 'within_budget' | 'over_budget' | 'close_to_exceed';
  usedBudget?: number;
  // Azure credentials
  deploymentName?: string;
  targetUri?: string;
  apiKey?: string;
  // AWS Bedrock credentials
  secretKey?: string;
  accessKey?: string;
  // Embedding model credentials
  embeddingModelApiKey?: string;
  // Embedding AWS Bedrock credentials
  embeddingAccessKey?: string;
  embeddingSecretKey?: string;
  // Embedding Azure credentials
  embeddingDeploymentName?: string;
  embeddingTargetUri?: string;
  embeddingAzureApiKey?: string;
}

export interface LLMConnectionsResponse {
  data: LLMConnection[];

}

export interface BudgetStatus {
  used_budget_percentage: number;
  exceeded_stop_budget: boolean;
  exceeded_warn_budget: boolean;
  data: {
    id: number;
    connectionName: string;
    usedBudget: number;
    monthlyBudget: number;
    warnBudgetThreshold: number;
    stopBudgetThreshold: number;
    environment: string;
    connectionStatus: string;
    createdAt: string;
    llmPlatform: string;
    llmModel: string;
    embeddingPlatform: string;
    embeddingModel: string;
  }
}

export interface LLMConnectionFilters {
  pageNumber?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: string;
  llmPlatform?: string;
  llmModel?: string;
  embeddingPlatform?: string;
  environment?: string;
  status?: string;
}

export interface ProductionConnectionFilters {
  llmPlatform?: string;
  llmModel?: string;
  embeddingPlatform?: string;
  embeddingModel?: string;
  connectionStatus?: string;
  sortBy?: string;
  sortOrder?: string;
}
export interface LegacyLLMConnectionFilters {
  page: number;
  pageSize: number;
  sorting?: string;
  llmPlatform?: string;
  embeddingPlatform?: string;
  environment?: string;
  status?: string;
}
export interface LLMConnectionFormData {
  connectionName: string;
  llmPlatform: string;
  llmModel: string;
  embeddingModelPlatform: string;
  embeddingModel: string;
  monthlyBudget: string;
  warnBudget: string;
  stopBudget: string;
  disconnectOnBudgetExceed: boolean;
  deploymentEnvironment: string;
  // Azure credentials
  deploymentName?: string;
  targetUri?: string;
  apiKey?: string;
  // AWS Bedrock credentials
  secretKey?: string;
  accessKey?: string;
  // Embedding model credentials
  embeddingModelApiKey?: string;
  // Embedding AWS Bedrock credentials
  embeddingAccessKey?: string;
  embeddingSecretKey?: string;
  // Embedding Azure credentials
  embeddingDeploymentName?: string;
  embeddingTargetUri?: string;
  embeddingAzureApiKey?: string;
}

// Vault secret service functions
async function createVaultSecret(vaultUuid: string, connectionData: LLMConnectionFormData): Promise<void> {

  // Encrypt sensitive credentials before sending to vault
  const encryptedCredentials = await encryptLLMCredentials({
    // AWS credentials
    secretKey: connectionData.secretKey,
    accessKey: connectionData.accessKey,
    // Azure credentials
    apiKey: connectionData.apiKey,
    // Embedding AWS credentials
    embeddingAccessKey: connectionData.embeddingAccessKey,
    embeddingSecretKey: connectionData.embeddingSecretKey,
    // Embedding Azure credentials
    embeddingAzureApiKey: connectionData.embeddingAzureApiKey,
  });

  const payload = {
    vaultUuid,
    llmPlatform: connectionData.llmPlatform,
    llmModel: connectionData.llmModel,
    embeddingModel: connectionData.embeddingModel,
    embeddingPlatform: connectionData.embeddingModelPlatform,
    deploymentEnvironment: connectionData.deploymentEnvironment.toLowerCase(),
    // AWS credentials (encrypted)
    ...(connectionData.llmPlatform === 'aws' && {
      secretKey: encryptedCredentials.secretKey || '',
      accessKey: encryptedCredentials.accessKey || '',
    }),
    // Azure credentials (encrypted)
    ...(connectionData.llmPlatform === 'azure' && {
      deploymentName: connectionData.deploymentName || '',
      targetUrl: connectionData.targetUri || '',
      apiKey: encryptedCredentials.apiKey || '',
    }),
    // Embedding AWS Bedrock credentials (encrypted)
    ...(connectionData.embeddingModelPlatform === 'aws' && {
      embeddingAccessKey: encryptedCredentials.embeddingAccessKey || '',
      embeddingSecretKey: encryptedCredentials.embeddingSecretKey || '',
    }),
    // Embedding Azure credentials (encrypted)
    ...(connectionData.embeddingModelPlatform === 'azure' && {
      embeddingDeploymentName: connectionData.embeddingDeploymentName || '',
      embeddingTargetUri: connectionData.embeddingTargetUri || '',
      embeddingAzureApiKey: encryptedCredentials.embeddingAzureApiKey || '',
    }),
  };

  await apiDev.post(vaultEndpoints.CREATE_VAULT_SECRET(), payload);
}

async function deleteVaultSecret(vaultUuid: string, connectionData: Partial<LLMConnectionFormData>): Promise<void> {

  const payload = {
    vaultUuid,
    llmPlatform: connectionData.llmPlatform || '',
    llmModel: connectionData.llmModel || '',
    embeddingModel: connectionData.embeddingModel || '',
    embeddingPlatform: connectionData.embeddingModelPlatform || '',
  };

  await apiDev.post(vaultEndpoints.DELETE_VAULT_SECRET(), payload);
}

export async function fetchLLMConnectionsPaginated(filters: LLMConnectionFilters): Promise<LLMConnection[]> {
  const queryParams = new URLSearchParams();

  if (filters.pageNumber) queryParams.append('pageNumber', filters.pageNumber.toString());
  if (filters.pageSize) queryParams.append('pageSize', filters.pageSize.toString());
  if (filters.sortBy) queryParams.append('sortBy', filters.sortBy);
  if (filters.sortOrder) queryParams.append('sortOrder', filters.sortOrder);
  if (filters.llmPlatform) queryParams.append('llmPlatform', filters.llmPlatform);
  if (filters.llmModel) queryParams.append('llmModel', filters.llmModel);
  if (filters.environment) queryParams.append('environment', filters.environment);

  const url = `${llmConnectionsEndpoints.FETCH_LLM_CONNECTIONS_PAGINATED()}?${queryParams.toString()}`;
  const { data } = await apiDev.get(url);
  return data?.response;
}

export async function getLLMConnection(id: string | number): Promise<LLMConnection> {
  const { data } = await apiDev.post(llmConnectionsEndpoints.GET_LLM_CONNECTION(), {
    connection_id: id,
  });
  return data?.response;
}

export async function getProductionConnection(filters?: ProductionConnectionFilters): Promise<LLMConnection | null> {
  const queryParams = new URLSearchParams();

  if (filters?.llmPlatform) queryParams.append('llmPlatform', filters.llmPlatform);
  if (filters?.llmModel) queryParams.append('llmModel', filters.llmModel);
  if (filters?.embeddingPlatform) queryParams.append('embeddingPlatform', filters.embeddingPlatform);
  if (filters?.embeddingModel) queryParams.append('embeddingModel', filters.embeddingModel);
  if (filters?.connectionStatus) queryParams.append('connectionStatus', filters.connectionStatus);
  if (filters?.sortBy) queryParams.append('sortBy', filters.sortBy);
  if (filters?.sortOrder) queryParams.append('sortOrder', filters.sortOrder);

  const url = queryParams.toString()
    ? `${llmConnectionsEndpoints.GET_PRODUCTION_CONNECTION()}?${queryParams.toString()}`
    : llmConnectionsEndpoints.GET_PRODUCTION_CONNECTION();

  const { data } = await apiDev.get(url);
  return data?.response?.[0] || null;
}


export async function createLLMConnection(connectionData: LLMConnectionFormData): Promise<LLMConnection> {
  const { data } = await apiDev.post(llmConnectionsEndpoints.CREATE_LLM_CONNECTION(), {
    connection_name: connectionData.connectionName,
    llm_platform: connectionData.llmPlatform,
    llm_model: connectionData.llmModel,
    embedding_platform: connectionData.embeddingModelPlatform,
    embedding_model: connectionData.embeddingModel,
    monthly_budget: parseFloat(removeCommasFromNumber(connectionData.monthlyBudget)),
    warn_budget_threshold: parseInt(connectionData.warnBudget),
    stop_budget_threshold: connectionData.disconnectOnBudgetExceed ? parseInt(connectionData.stopBudget) : 0,
    disconnect_on_budget_exceed: connectionData.disconnectOnBudgetExceed,
    deployment_environment: connectionData.deploymentEnvironment.toLowerCase(),
    // Azure credentials
    deployment_name: connectionData.deploymentName || "",
    target_uri: connectionData.targetUri || "",
    api_key: maskSensitiveKey(connectionData.apiKey) || "",
    // AWS Bedrock credentials
    secret_key: maskSensitiveKey(connectionData.secretKey) || "",
    access_key: maskSensitiveKey(connectionData.accessKey) || "",
    // Embedding model credentials
    // Embedding AWS Bedrock credentials
    embedding_access_key: maskSensitiveKey(connectionData.embeddingAccessKey) || "",
    embedding_secret_key: maskSensitiveKey(connectionData.embeddingSecretKey) || "",
    // Embedding Azure credentials
    embedding_deployment_name: connectionData.embeddingDeploymentName || "",
    embedding_target_uri: connectionData.embeddingTargetUri || "",
    embedding_azure_api_key: maskSensitiveKey(connectionData.embeddingAzureApiKey) || "",
  });

  const connection = data?.response;
  console.log('Created LLM Connection:', connection);

  // After successful database creation, store secrets in vault
  if (connection && connection.id && connection.vaultUuid) {
    await createVaultSecret(connection.vaultUuid, connectionData);
  }

  return connection;
}

export async function updateLLMConnection(
  id: string | number,
  connectionData: LLMConnectionFormData
): Promise<LLMConnection> {
  const { data } = await apiDev.post(llmConnectionsEndpoints.UPDATE_LLM_CONNECTION(), {
    connection_id: id,
    connection_name: connectionData.connectionName,
    llm_platform: connectionData.llmPlatform,
    llm_model: connectionData.llmModel,
    embedding_platform: connectionData.embeddingModelPlatform,
    embedding_model: connectionData.embeddingModel,
    monthly_budget: parseFloat(removeCommasFromNumber(connectionData.monthlyBudget)),
    warn_budget_threshold: parseInt(connectionData.warnBudget),
    stop_budget_threshold: connectionData.disconnectOnBudgetExceed ? parseInt(connectionData.stopBudget) : 0,
    disconnect_on_budget_exceed: connectionData.disconnectOnBudgetExceed,
    deployment_environment: connectionData.deploymentEnvironment.toLowerCase(),
    // Azure credentials
    deployment_name: connectionData.deploymentName || "",
    target_uri: connectionData.targetUri || "",
    api_key: maskSensitiveKey(connectionData.apiKey) || "",
    // AWS Bedrock credentials
    secret_key: maskSensitiveKey(connectionData.secretKey) || "",
    access_key: maskSensitiveKey(connectionData.accessKey) || "",
    // Embedding model credentials
    // Embedding AWS Bedrock credentials
    embedding_access_key: maskSensitiveKey(connectionData.embeddingAccessKey) || "",
    embedding_secret_key: maskSensitiveKey(connectionData.embeddingSecretKey) || "",
    // Embedding Azure credentials
    embedding_deployment_name: connectionData.embeddingDeploymentName || "",
    embedding_target_uri: connectionData.embeddingTargetUri || "",
    embedding_azure_api_key: maskSensitiveKey(connectionData.embeddingAzureApiKey) || "",
  });

  const connection = data?.response;

  if (connection && (connectionData.secretKey && !connectionData.secretKey?.includes('*')
    || connectionData.accessKey && !connectionData.accessKey?.includes('*')
    || connectionData.apiKey && !connectionData.apiKey?.includes('*')
    || connectionData.embeddingAccessKey && !connectionData.embeddingAccessKey?.includes('*')
    || connectionData.embeddingSecretKey && !connectionData.embeddingSecretKey?.includes('*')
    || connectionData.embeddingAzureApiKey && !connectionData.embeddingAzureApiKey?.includes('*'))) {
    try {
      const vaultUuid = connection.vaultUuid || (await getLLMConnection(id)).vaultUuid;
      if (vaultUuid) {
        await createVaultSecret(vaultUuid, connectionData);
      }
    } catch (vaultError) {
      console.error('Failed to update secrets in vault:', vaultError);
    }
  }

  return connection;
}

export async function deleteLLMConnection(id: string | number): Promise<void> {
  // First, get the connection data to extract vault deletion parameters
  let connectionToDelete: LLMConnection | null = null;
  try {
    connectionToDelete = await getLLMConnection(id);
  } catch (error) {
    console.error('Failed to get connection data before deletion:', error);
  }

  // Delete secrets from vault BEFORE database deletion
  // (delete.yml validates connection exists, so DB must not be soft-deleted yet)
  if (connectionToDelete && connectionToDelete.vaultUuid) {
    try {
      await deleteVaultSecret(connectionToDelete.vaultUuid, {
        llmPlatform: connectionToDelete.llmPlatform,
        llmModel: connectionToDelete.llmModel,
        embeddingModel: connectionToDelete.embeddingModel,
        embeddingModelPlatform: connectionToDelete.embeddingPlatform,
      });
    } catch (vaultError) {
      console.error('Failed to delete secrets from vault:', vaultError);
      // Continue with database deletion even if vault deletion fails
    }
  }

  // Delete from database (soft-delete sets connection_status = 'deleted')
  await apiDev.post(llmConnectionsEndpoints.DELETE_LLM_CONNECTION(), {
    connection_id: id,
  });
}

export async function checkBudgetStatus(): Promise<BudgetStatus | null> {
  try {
    const { data } = await apiDev.get(llmConnectionsEndpoints.CHECK_BUDGET_STATUS());
    return data?.response as BudgetStatus;
  } catch (error) {
    // Return null if no production connection found (404) or other errors
    return null;
  }
}

export async function updateLLMConnectionStatus(
  id: string | number,
  status: 'active' | 'inactive'
): Promise<LLMConnection> {
  const { data } = await apiDev.post(llmConnectionsEndpoints.UPDATE_LLM_CONNECTION_STATUS(), {
    connection_id: id,
    connection_status: status,
  });
  return data?.response;
}

export async function fetchAllLLMConnectionsPaginated(filters: LLMConnectionFilters): Promise<LLMConnection[]> {
  const queryParams = new URLSearchParams();

  if (filters.pageNumber) queryParams.append('pageNumber', filters.pageNumber.toString());
  if (filters.pageSize) queryParams.append('pageSize', filters.pageSize.toString());
  if (filters.sortBy) queryParams.append('sortBy', filters.sortBy);
  if (filters.sortOrder) queryParams.append('sortOrder', filters.sortOrder);
  if (filters.llmPlatform) queryParams.append('llmPlatform', filters.llmPlatform);
  if (filters.llmModel) queryParams.append('llmModel', filters.llmModel);
  if (filters.environment) queryParams.append('environment', filters.environment);

  const url = `${llmConnectionsEndpoints.FETCH_ALL_LLM_CONNECTIONS_PAGINATED()}?${queryParams.toString()}`;
  const { data } = await apiDev.get(url);
  return data?.response;
}
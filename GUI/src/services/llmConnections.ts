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


/**
 * The connection's embedding model was refused (issue #806). `code` picks the
 * translated message: 'model_mismatch' (shown with `requiredModel`) or
 * 'test_embedding_failed'.
 */
export class EmbeddingModelLockedError extends Error {
  readonly code: string;
  // 'different_model' | 'wrong_deployment' (for code 'model_mismatch')
  readonly detail: string | null;
  readonly requiredModel: string | null;
  readonly requestedModel: string | null;
  readonly detectedDimensions: number | null;

  constructor(result: EmbeddingCheckResult) {
    super(result.reason || 'The embedding model is not allowed');
    this.name = 'EmbeddingModelLockedError';
    this.code = result.code || 'model_mismatch';
    this.detail = result.detail ?? null;
    this.requiredModel = result.required_model ?? null;
    this.requestedModel = result.requested_model ?? null;
    this.detectedDimensions = result.detected_vector_dimensions ?? null;
  }
}

/** The embedding model could not be checked (#806), so nothing was saved. */
export class EmbeddingCheckUnavailableError extends Error {
  readonly originalError: unknown;

  constructor(originalError: unknown) {
    super('The embedding model could not be checked against the knowledge base');
    this.name = 'EmbeddingCheckUnavailableError';
    this.originalError = originalError;
  }
}

interface EmbeddingCheckResult {
  allowed: boolean;
  code?: string | null;
  detail?: string | null;
  required_model?: string | null;
  requested_model?: string | null;
  detected_vector_dimensions?: number | null;
  reason?: string | null;
}

// A blocked save comes back as a 400 response with { allowed: false, code, ... }.
// A check that could not run is flagged check_unavailable and is not a lock.
function embeddingLockFromError(error: unknown): EmbeddingModelLockedError | null {
  const data = (error as { response?: { data?: any } })?.response?.data;
  const body: EmbeddingCheckResult & { check_unavailable?: boolean } = data?.response ?? data;
  return body && body.allowed === false && !body.check_unavailable
    ? toLockedError(body)
    : null;
}

function toLockedError(result: EmbeddingCheckResult): EmbeddingModelLockedError {
  return new EmbeddingModelLockedError(result);
}

function newVaultUuid(): string {
  if (typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  // RFC 4122 version 4 from random bytes, for browsers without randomUUID
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

// A key typed in the form, as opposed to the masked saved value ("****")
function isNewSecret(value?: string): boolean {
  return !!value && !value.includes('*');
}

async function postEmbeddingCheck(body: Record<string, unknown>): Promise<EmbeddingCheckResult> {
  try {
    const { data } = await apiDev.post(llmConnectionsEndpoints.CHECK_EMBEDDING(), body);
    return data?.response ?? data;
  } catch (error) {
    const locked = embeddingLockFromError(error);
    if (locked) {
      throw locked;
    }
    throw error;
  }
}

/**
 * Check the embedding model really behind the entered credentials against the
 * knowledge base (issue #806) before anything is saved.
 *
 * - New embedding credentials: stored in Vault under a temporary id, tested with a
 *   test embedding, and removed again by the server.
 * - Saved key kept: the saved credentials are tested with the entered Azure
 *   deployment name and endpoint.
 * - Nothing embedding-related changed (edit only): no check.
 *
 * Throws EmbeddingModelLockedError when the model is not allowed, and
 * EmbeddingCheckUnavailableError when the check could not be made.
 */
async function verifyEmbeddingModel(
  connectionData: LLMConnectionFormData,
  existing?: LLMConnection
): Promise<void> {
  const embeddingKeyEntered =
    isNewSecret(connectionData.embeddingAzureApiKey) ||
    isNewSecret(connectionData.embeddingAccessKey) ||
    isNewSecret(connectionData.embeddingSecretKey);

  const body: Record<string, unknown> = {
    embedding_platform: connectionData.embeddingModelPlatform,
    embedding_model: connectionData.embeddingModel,
    embedding_deployment_name: connectionData.embeddingDeploymentName || '',
    embedding_target_uri: connectionData.embeddingTargetUri || '',
    // #588: production may use another approved model; the knowledge base is then re-indexed
    deployment_environment: connectionData.deploymentEnvironment.toLowerCase(),
    ...(existing && { exclude_connection_id: Number(existing.id) }),
  };

  const useTemporarySecret = !existing || embeddingKeyEntered;
  if (useTemporarySecret) {
    body.vault_uuid = newVaultUuid();
    body.llm_platform = connectionData.llmPlatform;
    body.llm_model = connectionData.llmModel;
  } else {
    const embeddingChanged =
      existing.embeddingPlatform !== connectionData.embeddingModelPlatform ||
      existing.embeddingModel !== connectionData.embeddingModel ||
      (existing.embeddingDeploymentName || '') !== (connectionData.embeddingDeploymentName || '') ||
      (existing.embeddingTargetUri || '') !== (connectionData.embeddingTargetUri || '');
    if (!embeddingChanged) {
      return;
    }
    // The saved key is kept, so test the saved credentials with the new deployment/endpoint
    if (existing.vaultUuid && connectionData.embeddingModelPlatform === existing.embeddingPlatform) {
      body.existing_vault_uuid = existing.vaultUuid;
    }
  }

  let result: EmbeddingCheckResult;
  try {
    if (useTemporarySecret) {
      await createVaultSecret(body.vault_uuid as string, connectionData);
    }
    result = await postEmbeddingCheck(body);
  } catch (error) {
    if (error instanceof EmbeddingModelLockedError) {
      throw error;
    }
    throw new EmbeddingCheckUnavailableError(error);
  }
  if (!result?.allowed) {
    throw toLockedError(result);
  }
}

/**
 * Step of createLLMConnection() that failed:
 * - embeddingLocked: the embedding model conflicts with the knowledge base; nothing was saved
 * - embeddingCheck: the embedding model could not be checked; nothing was saved
 * - save: the connection could not be saved; nothing was stored
 * - vault: credentials could not be stored in Vault; the saved connection was removed again
 * - vaultRollbackFailed: credentials could not be stored and the saved connection could not be removed
 * - promote: the connection was created as testing but could not be switched to production
 */
export type CreateLLMConnectionFailureStage =
  | 'embeddingLocked'
  | 'embeddingCheck'
  | 'save'
  | 'vault'
  | 'vaultRollbackFailed'
  | 'promote';

export class CreateLLMConnectionError extends Error {
  readonly stage: CreateLLMConnectionFailureStage;
  readonly originalError: unknown;
  // Only set for the 'promote' stage: true when no production connection is active afterwards
  readonly noProductionConnection: boolean;

  constructor(stage: CreateLLMConnectionFailureStage, originalError: unknown, noProductionConnection = false) {
    super(`Creating LLM connection failed at stage: ${stage}`);
    this.name = 'CreateLLMConnectionError';
    this.stage = stage;
    this.originalError = originalError;
    this.noProductionConnection = noProductionConnection;
  }
}

// Request body shared by the create (/llm-connections/add) and edit (/llm-connections/edit) endpoints
function buildLLMConnectionPayload(connectionData: LLMConnectionFormData, deploymentEnvironment: string) {
  return {
    connection_name: connectionData.connectionName,
    llm_platform: connectionData.llmPlatform,
    llm_model: connectionData.llmModel,
    embedding_platform: connectionData.embeddingModelPlatform,
    embedding_model: connectionData.embeddingModel,
    monthly_budget: parseFloat(removeCommasFromNumber(connectionData.monthlyBudget)),
    warn_budget_threshold: parseInt(connectionData.warnBudget),
    stop_budget_threshold: connectionData.disconnectOnBudgetExceed ? parseInt(connectionData.stopBudget) : 0,
    disconnect_on_budget_exceed: connectionData.disconnectOnBudgetExceed,
    deployment_environment: deploymentEnvironment.toLowerCase(),
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
  };
}

/**
 * Create an LLM connection without leaving a half-created connection behind.
 *
 * 1. Save the connection as testing, even when production was chosen, so the
 *    current production connection is not touched yet.
 * 2. Store the credentials in Vault. If that fails, delete the saved connection.
 * 3. Only if production was chosen, switch the connection to production.
 *    /llm-connections/edit then moves the previous production connection to testing.
 *
 * Throws CreateLLMConnectionError with the step that failed.
 */
export async function createLLMConnection(connectionData: LLMConnectionFormData): Promise<LLMConnection> {
  const wantsProduction = connectionData.deploymentEnvironment.toLowerCase() === 'production';

  // 0. Check the real embedding model against the knowledge base (#806); nothing is saved yet
  try {
    await verifyEmbeddingModel(connectionData);
  } catch (checkError) {
    throw new CreateLLMConnectionError(
      checkError instanceof EmbeddingModelLockedError ? 'embeddingLocked' : 'embeddingCheck',
      checkError
    );
  }

  // 1. Save as testing
  let connection: LLMConnection | undefined;
  try {
    const { data } = await apiDev.post(llmConnectionsEndpoints.CREATE_LLM_CONNECTION(), {
      ...buildLLMConnectionPayload(connectionData, 'testing'),
      // Saved as testing but checked as the environment it is promoted to in step 3:
      // production may use another embedding model than the knowledge base (#588)
      target_environment: wantsProduction ? 'production' : 'testing',
    });
    connection = data?.response;
  } catch (saveError) {
    // The server also checks the embedding model (in case the step above was bypassed)
    const locked = embeddingLockFromError(saveError);
    if (locked) {
      throw new CreateLLMConnectionError('embeddingLocked', locked);
    }
    throw new CreateLLMConnectionError('save', saveError);
  }

  if (!connection?.id) {
    throw new CreateLLMConnectionError('save', new Error('Create response did not contain a connection id'));
  }
  const connectionId = Number(connection.id);

  // 2. Store credentials in Vault; on failure remove the saved connection again
  try {
    if (!connection.vaultUuid) {
      throw new Error('Create response did not contain a vaultUuid');
    }
    await createVaultSecret(connection.vaultUuid, connectionData);
  } catch (vaultError) {
    console.error('Storing LLM connection credentials failed, removing the saved connection:', vaultError);
    try {
      await apiDev.post(llmConnectionsEndpoints.DELETE_LLM_CONNECTION(), { connection_id: connectionId });
    } catch (rollbackError) {
      console.error('Removing the saved LLM connection failed:', rollbackError);
      throw new CreateLLMConnectionError('vaultRollbackFailed', vaultError);
    }
    throw new CreateLLMConnectionError('vault', vaultError);
  }

  // 3. Switch to production only after the credentials are stored
  if (wantsProduction) {
    try {
      const { data } = await apiDev.post(llmConnectionsEndpoints.UPDATE_LLM_CONNECTION(), {
        connection_id: connectionId,
        ...buildLLMConnectionPayload(connectionData, 'production'),
      });
      connection = data?.response ?? connection;
    } catch (promoteError) {
      console.error('Switching the new LLM connection to production failed:', promoteError);
      let noProductionConnection = false;
      try {
        noProductionConnection = !(await getProductionConnection());
      } catch (checkError) {
        console.error('Checking for an active production connection failed:', checkError);
      }
      throw new CreateLLMConnectionError('promote', promoteError, noProductionConnection);
    }
  }

  return connection as LLMConnection;
}

/**
 * Update an LLM connection. When an embedding setting changed, the real embedding
 * model is checked against the knowledge base first (#806); a conflict throws
 * EmbeddingModelLockedError and nothing is changed.
 */
export async function updateLLMConnection(
  id: string | number,
  connectionData: LLMConnectionFormData
): Promise<LLMConnection> {
  const existing = await getLLMConnection(id);
  await verifyEmbeddingModel(connectionData, existing);

  let data;
  try {
    ({ data } = await apiDev.post(llmConnectionsEndpoints.UPDATE_LLM_CONNECTION(), {
      connection_id: id,
      ...buildLLMConnectionPayload(connectionData, connectionData.deploymentEnvironment),
    }));
  } catch (updateError) {
    // The server also checks the embedding model (in case the step above was bypassed)
    const locked = embeddingLockFromError(updateError);
    if (locked) {
      throw locked;
    }
    throw updateError;
  }

  const connection = data?.response;

  if (connection && (connectionData.secretKey && !connectionData.secretKey?.includes('*')
    || connectionData.accessKey && !connectionData.accessKey?.includes('*')
    || connectionData.apiKey && !connectionData.apiKey?.includes('*')
    || connectionData.embeddingAccessKey && !connectionData.embeddingAccessKey?.includes('*')
    || connectionData.embeddingSecretKey && !connectionData.embeddingSecretKey?.includes('*')
    || connectionData.embeddingAzureApiKey && !connectionData.embeddingAzureApiKey?.includes('*'))) {
    try {
      const vaultUuid = connection.vaultUuid || existing.vaultUuid;
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

/**
 * Activate or deactivate a connection. Activating an inactive connection first
 * checks its saved embedding model and credentials against the knowledge base
 * (#806): throws EmbeddingModelLockedError when it is not allowed and
 * EmbeddingCheckUnavailableError when it could not be checked.
 *
 * The check goes through check-embedding (200 with { allowed, ... }) because the
 * api-dev interceptor drops the body of error responses, so the 400 that
 * update-status returns as a server-side backstop carries no readable reason.
 */
export async function updateLLMConnectionStatus(
  id: string | number,
  status: 'active' | 'inactive'
): Promise<LLMConnection> {
  if (status === 'active') {
    const existing = await getLLMConnection(id);
    if (existing && existing.connectionStatus !== 'active') {
      let result: EmbeddingCheckResult;
      try {
        result = await postEmbeddingCheck({
          embedding_platform: existing.embeddingPlatform,
          embedding_model: existing.embeddingModel,
          existing_vault_uuid: existing.vaultUuid || '',
          exclude_connection_id: Number(existing.id),
        });
      } catch (error) {
        if (error instanceof EmbeddingModelLockedError) {
          throw error;
        }
        throw new EmbeddingCheckUnavailableError(error);
      }
      if (!result?.allowed) {
        throw toLockedError(result);
      }
    }
  }

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
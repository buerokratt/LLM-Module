import {
  EmbeddingCheckUnavailableError,
  EmbeddingModelLockedError,
  KnowledgeBaseReindexNotice,
} from 'services/llmConnections';

type Translate = (key: string, options?: Record<string, unknown>) => string;

// What happened to the user's action: create, edit, or activate (toggle)
export type EmbeddingLockOutcome = 'notSaved' | 'notChanged' | 'notActivated';

/**
 * One short message for an embedding model refusal (issue #806): the specific
 * cause, then what happened, e.g. "This connection uses amazon.titan-embed-text-v2:0,
 * but all connections must use text-embedding-3-large. The connection was not saved."
 *
 * Returns null when the error is not an embedding model refusal.
 */
export function embeddingLockMessage(
  t: Translate,
  error: unknown,
  outcome: EmbeddingLockOutcome
): string | null {
  let cause: string;
  if (error instanceof EmbeddingModelLockedError) {
    if (error.code === 'test_embedding_failed' && error.detail === 'endpoint_host_changed') {
      // The saved key is never sent to a new host; it must be entered again
      cause = t('llmConnectionForm.embeddingLock.endpointHostChanged');
    } else if (error.code === 'test_embedding_failed') {
      cause = t('llmConnectionForm.embeddingLock.testFailed');
    } else if (error.detail === 'wrong_deployment') {
      cause = t('llmConnectionForm.embeddingLock.wrongDeployment', {
        model: error.requiredModel ?? '',
        dims: error.detectedDimensions ?? '',
      });
    } else {
      // Say why: the model of the existing embeddings, or (knowledge base still
      // empty) of the other active connection, which the first index will use
      cause = t(
        error.lockedBy === 'connection'
          ? 'llmConnectionForm.embeddingLock.differentModelConnection'
          : 'llmConnectionForm.embeddingLock.differentModel',
        {
          current: error.requestedModel ?? '',
          model: error.requiredModel ?? '',
        }
      );
    }
  } else if (error instanceof EmbeddingCheckUnavailableError) {
    cause = t('llmConnectionForm.embeddingLock.unavailable');
  } else {
    return null;
  }
  return `${cause} ${t(`llmConnectionForm.embeddingLock.${outcome}`)}`;
}

/**
 * What a production connection with another embedding model changes (#588):
 * the knowledge base is re-indexed with its model, and other connections still
 * on the old model find nothing until they are switched too. One line each.
 */
export function knowledgeBaseReindexMessages(
  t: Translate,
  notice: KnowledgeBaseReindexNotice | null
): string[] {
  if (!notice) {
    return [];
  }
  return [
    t('llmConnectionForm.knowledgeBaseReindex.started', {
      from: notice.fromModels.join(', '),
      to: notice.toModel,
    }),
    ...notice.mismatchedConnections.map((other) =>
      t('llmConnectionForm.knowledgeBaseReindex.connectionMismatch', {
        name: other.name,
        environment: other.environment,
        model: other.model,
        to: notice.toModel,
      })
    ),
  ];
}

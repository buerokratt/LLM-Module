import { EmbeddingCheckUnavailableError, EmbeddingModelLockedError } from 'services/llmConnections';

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
    if (error.code === 'test_embedding_failed') {
      cause = t('llmConnectionForm.embeddingLock.testFailed');
    } else if (error.detail === 'wrong_deployment') {
      cause = t('llmConnectionForm.embeddingLock.wrongDeployment', {
        model: error.requiredModel ?? '',
        dims: error.detectedDimensions ?? '',
      });
    } else {
      cause = t('llmConnectionForm.embeddingLock.differentModel', {
        current: error.requestedModel ?? '',
        model: error.requiredModel ?? '',
      });
    }
  } else if (error instanceof EmbeddingCheckUnavailableError) {
    cause = t('llmConnectionForm.embeddingLock.unavailable');
  } else {
    return null;
  }
  return `${cause} ${t(`llmConnectionForm.embeddingLock.${outcome}`)}`;
}

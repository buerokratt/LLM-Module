import apiDev from './api';
import { inferenceEndpoints } from 'utils/endpoints';

export interface InferenceRequest {
  llmConnectionId: number;
  message: string;
}

// Remove after testing
export interface ProductionInferenceRequest {
  chatId: string;
  message: string;
  authorId: string;
  conversationHistory: Array<{
    authorRole: string;
    message: string;
    timestamp: string;
  }>;
  url: string;
}

export interface ChoiceButton {
  title: string;
  payload: string;
}

export interface InferenceResponse {
  response: {
    chatId: number;
    llmServiceActive: boolean;
    questionOutOfLlmScope: boolean;
    content: string;
    buttons?: ChoiceButton[];
    chunks?: {
      rank: number,
      chunkRetrieved: string
    }[]
  };
}

// Remove after testing
export interface ProductionInferenceResponse {
  chatId: string;
  content: string;
  llmServiceActive?: boolean;
  questionOutOfLlmScope?: boolean;
  status?: number;
}

export async function viewInferenceResult(request: InferenceRequest): Promise<InferenceResponse> {
  const { data } = await apiDev.post(inferenceEndpoints.VIEW_TEST_INFERENCE_RESULT(), {
    connectionId: request.llmConnectionId,
    message: request.message,
  });
  return data;
}

// Remove after testing
export async function productionInference(request: ProductionInferenceRequest): Promise<ProductionInferenceResponse> {
  try {
    const { data } = await apiDev.post(inferenceEndpoints.PRODUCTION_INFERENCE(), request);
    return data;
  } catch (error: any) {
    // Handle error responses
    if (error.response?.data) {
      return error.response.data;
    }
    throw error;
  }
}

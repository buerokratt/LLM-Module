import apiDev from './api';
import { promptConfigurationEndpoints } from 'utils/endpoints';

export interface PromptConfiguration {
    id: number | null;
    prompt: string;
}

export interface PromptConfigurationResponse {
    response: PromptConfiguration[];
}

export const getPromptConfiguration = async (): Promise<PromptConfiguration[]> => {
    const { data } = await apiDev.get(promptConfigurationEndpoints.GET_PROMPT_CONFIGURATION());
    return data?.response || [];
};

export const savePromptConfiguration = async (prompt: string): Promise<PromptConfiguration[]> => {
    const { data } = await apiDev.post(promptConfigurationEndpoints.SAVE_PROMPT_CONFIGURATION(), {
        prompt,
    });
    return data?.response;
};

export const disablePromptConfiguration = async (): Promise<PromptConfiguration[]> => {
    const { data } = await apiDev.post(promptConfigurationEndpoints.SAVE_PROMPT_CONFIGURATION(), {
        prompt: "",
    });
    return data?.response;
};
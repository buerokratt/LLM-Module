import React, { useEffect, useState } from 'react';
import { useForm, Controller } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';
import FormInput from 'components/FormElements/FormInput';
import FormSelect from 'components/FormElements/FormSelect';
import FormCheckbox from 'components/FormElements/FormCheckbox';
import Button from 'components/Button';
import Track from 'components/Track';
import {
  getLLMPlatforms,
  getLLMModels,
  getEmbeddingPlatforms,
  getEmbeddingModels,
  PlatformOption,
  ModelOption
} from 'services/llmConfigs';
import './LLMConnectionForm.scss';
import { toOptions } from 'utils/commonUtils';

export type LLMConnectionFormData = {
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
  // AWS Bedrock credentials
  accessKey?: string;
  awsRegion?: string;
  secretKey?: string;
  // Azure credentials
  deploymentName?: string;
  targetUri?: string;
  apiKey?: string;
  // Embedding model credentials
  embeddingModelApiKey?: string;
  // Embedding AWS Bedrock credentials
  embeddingAccessKey?: string;
  embeddingAwsRegion?: string;
  embeddingSecretKey?: string;
  // Embedding Azure credentials
  embeddingDeploymentName?: string;
  embeddingTargetUri?: string;
  embeddingAzureApiKey?: string;
};

type LLMConnectionFormProps = {
  onSubmit: (data: LLMConnectionFormData) => void;
  onCancel: () => void;
  onDelete: () => void;
  defaultValues?: Partial<LLMConnectionFormData>;
  isEditing?: boolean;
  readOnly?: boolean;
};

const LLMConnectionForm: React.FC<LLMConnectionFormProps> = ({
  onSubmit,
  onCancel,
  onDelete,
  defaultValues,
  isEditing = false,
  readOnly = false,
}) => {
  const { t } = useTranslation();
  const {
    control,
    handleSubmit,
    watch,
    setValue,
    formState: { errors, isDirty, isValid },
  } = useForm<LLMConnectionFormData>({
    defaultValues: {
      connectionName: '',
      llmPlatform: '',
      llmModel: '',
      embeddingModelPlatform: '',
      embeddingModel: '',
      monthlyBudget: '',
      warnBudget: '',
      stopBudget: '',
      disconnectOnBudgetExceed: false,
      deploymentEnvironment: '',
      // AWS Bedrock credentials
      accessKey: '',
      awsRegion: '',
      secretKey: '',
      // Azure credentials
      deploymentName: '',
      targetUri: '',
      apiKey: '',
      // Embedding model credentials
      embeddingModelApiKey: '',
      // Embedding AWS Bedrock credentials
      embeddingAccessKey: '',
      embeddingAwsRegion: '',
      embeddingSecretKey: '',
      // Embedding Azure credentials
      embeddingDeploymentName: '',
      embeddingTargetUri: '',
      embeddingAzureApiKey: '',
      ...defaultValues,
    },
    mode: 'onChange',
  });

  const selectedLLMPlatform = watch('llmPlatform');
  const selectedEmbeddingPlatform = watch('embeddingModelPlatform');
  const disconnectOnBudgetExceed = watch('disconnectOnBudgetExceed');

  // Fetch platform and model options from API
  const { data: llmPlatformsData = [], isLoading: llmPlatformsLoading, error: llmPlatformsError } = useQuery({
    queryKey: ['llm-platforms'],
    queryFn: getLLMPlatforms
  });

  const { data: embeddingPlatformsData = [], isLoading: embeddingPlatformsLoading, error: embeddingPlatformsError } = useQuery({
    queryKey: ['embedding-platforms'],
    queryFn: getEmbeddingPlatforms
  });

  const { data: llmModelsData = [], isLoading: llmModelsLoading, error: llmModelsError } = useQuery({
    queryKey: ['llm-models', selectedLLMPlatform],
    queryFn: () => getLLMModels(selectedLLMPlatform),
    enabled: !!selectedLLMPlatform,
  });

  const { data: embeddingModelsData = [], isLoading: embeddingModelsLoading, error: embeddingModelsError } = useQuery({
    queryKey: ['embedding-models', selectedEmbeddingPlatform],
    queryFn: () => getEmbeddingModels(selectedEmbeddingPlatform),
    enabled: !!selectedEmbeddingPlatform,
  });

const llmPlatformOptions = toOptions(llmPlatformsData);
const embeddingPlatformOptions = toOptions(embeddingPlatformsData);
const llmModelOptions = toOptions(llmModelsData);
const embeddingModelOptions = toOptions(embeddingModelsData);

  const [apiKeyReplaceMode, setApiKeyReplaceMode] = React.useState(isEditing);
  const [secretKeyReplaceMode, setSecretKeyReplaceMode] = React.useState(isEditing);
  const [accessKeyReplaceMode, setAccessKeyReplaceMode] = React.useState(isEditing);
  const [embeddingApiKeyReplaceMode, setEmbeddingApiKeyReplaceMode] = React.useState(isEditing);
  // Embedding platform specific replace modes
  const [embeddingSecretKeyReplaceMode, setEmbeddingSecretKeyReplaceMode] = React.useState(isEditing);
  const [embeddingAccessKeyReplaceMode, setEmbeddingAccessKeyReplaceMode] = React.useState(isEditing);
  const [embeddingAzureApiKeyReplaceMode, setEmbeddingAzureApiKeyReplaceMode] = React.useState(isEditing);

  const resetLLMCredentialFields = () => {
    setValue('accessKey', '');
    setValue('secretKey', '');
    setValue('deploymentName', '');
    setValue('targetUri', '');
    setValue('apiKey', '');
    setValue('llmModel', '');

    // Reset replace mode states when platform changes
    setApiKeyReplaceMode(false);
    setSecretKeyReplaceMode(false);
    setAccessKeyReplaceMode(false);
  };

  const resetEmbeddingModelCredentialFields = () => {
    setValue('embeddingModelApiKey', '');
    setValue('embeddingModel', '');
    // Reset embedding platform specific fields
    setValue('embeddingAccessKey', '');
    setValue('embeddingSecretKey', '');
    setValue('embeddingDeploymentName', '');
    setValue('embeddingTargetUri', '');
    setValue('embeddingAzureApiKey', '');

    // Reset replace mode states when platform changes
    setEmbeddingApiKeyReplaceMode(false);
    setEmbeddingSecretKeyReplaceMode(false);
    setEmbeddingAccessKeyReplaceMode(false);
    setEmbeddingAzureApiKeyReplaceMode(false);
  };
  // Model options based on selected platform
  const getLLMModelOptions = () => {
    return llmModelOptions;
  };

  const getEmbeddingModelOptions = () => {
    return embeddingModelOptions;
  };

  const deploymentEnvironments = [
    { label: t('llmConnectionForm.environments.testing') || 'Testing', value: 'testing' },
    { label: t('llmConnectionForm.environments.production') || 'Production', value: 'production' },
  ];

  const renderPlatformSpecificFields = () => {
    switch (selectedLLMPlatform) {
      case 'aws':
        return (
          <>
            <div className="llm-connection-form__field"><p className="form-label">AWS region</p><Controller name="awsRegion" control={control} rules={{required: 'AWS region is required'}} render={({field}) => <FormInput {...field} label="AWS region" placeholder="e.g. us-east-1" error={errors.awsRegion?.message} readOnly={readOnly} />} /></div>
              <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.aws.accessKey.label') || 'Access Key'}</p>
              <p className='form-description'>{t('llmConnectionForm.aws.accessKey.description') || 'AWS Access Key for Bedrock service'}</p>
              <Controller
                name="accessKey"
                control={control}
                rules={{ 
                  required: t('llmConnectionForm.validationMessages.accessKeyRequiredAws') || 'Access Key is required for AWS Bedrock',
                  pattern: {
                    value: /^\S+$/,
                    message: t('llmConnectionForm.validationMessages.invalidAccessKey') || 'Access Key cannot contain spaces'
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    type={isEditing ? 'text' : 'password'}
                    placeholder={t('llmConnectionForm.aws.accessKey.placeholder') || "Enter AWS Access Key"}
                    error={errors.accessKey?.message}
                    readOnly={accessKeyReplaceMode}
                    showEndButton={accessKeyReplaceMode}
                    onEndButtonClick={() => {
                      setAccessKeyReplaceMode(false);
                      setValue('accessKey', '');
                    }}
                    endButtonText={t('global.change') || "Change"}
                    {...field}
                  />
                )}
              />
            </div>
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.aws.secretKey.label') || 'Secret Key'}</p>
              <p className='form-description'>{t('llmConnectionForm.aws.secretKey.description') || 'AWS Secret Key for Bedrock service'}</p>
              <Controller
                name="secretKey"
                control={control}
                rules={{ 
                  required: t('llmConnectionForm.validationMessages.secretKeyRequiredAws') || 'Secret Key is required for AWS Bedrock',
                  pattern: {
                    value: /^\S+$/,
                    message: t('llmConnectionForm.validationMessages.invalidSecretKey') || 'Secret Key cannot contain spaces'
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    type={isEditing ? 'text' : 'password'}
                    placeholder={t('llmConnectionForm.aws.secretKey.placeholder') || "Enter AWS Secret Key"}
                    error={errors.secretKey?.message}
                    readOnly={secretKeyReplaceMode}
                    showEndButton={secretKeyReplaceMode}
                    onEndButtonClick={() => {
                      setSecretKeyReplaceMode(false);
                      setValue('secretKey', '');
                    }}
                    endButtonText={t('global.change') || "Change"}
                    {...field}
                  />
                )}
              />
            </div>
          </>
        );
      case 'azure':
        return (
          <>
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.azure.deploymentName.label') || 'Deployment Name'}</p>
              <p className='form-description'>{t('llmConnectionForm.azure.deploymentName.description') || 'Azure OpenAI deployment name'}</p>
              <Controller
                name="deploymentName"
                control={control}
                rules={{ required: t('llmConnectionForm.validationMessages.deploymentNameRequiredAzure') || 'Deployment Name is required for Azure OpenAI' }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    placeholder={t('llmConnectionForm.azure.deploymentName.placeholder') || "Enter deployment name"}
                    error={errors.deploymentName?.message}
                    {...field}
                  />
                )}
              />
            </div>
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.azure.targetUri.label') || 'Endpoint / Target URI'}</p>
              <p className='form-description'>{t('llmConnectionForm.azure.targetUri.description') || 'Azure OpenAI service endpoint URL'}</p>
              <Controller
                name="targetUri"
                control={control}
                rules={{
                  required: t('llmConnectionForm.validationMessages.endpointRequiredAzure') || 'Endpoint is required for Azure OpenAI',
                  pattern: {
                    value: /^https?:\/\/.+/,
                    message: t('llmConnectionForm.validationMessages.invalidUrl') || 'Please enter a valid URL starting with http:// or https://'
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    placeholder={t('llmConnectionForm.azure.targetUri.placeholder') || "https://your-resource.openai.azure.com/"}
                    error={errors.targetUri?.message}
                    {...field}
                  />
                )}
              />
            </div>
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.azure.apiKey.label') || 'API Key'}</p>
              <p className='form-description'>{t('llmConnectionForm.azure.apiKey.description') || 'Azure OpenAI API key'}</p>

              <Controller
                name="apiKey"
                control={control}
                rules={{ 
                  required: t('llmConnectionForm.validationMessages.apiKeyRequiredAzure') || 'API Key is required for Azure OpenAI',
                  pattern: {
                    value: /^\S+$/,
                    message: t('llmConnectionForm.validationMessages.invalidApiKey') || 'API Key cannot contain spaces'
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    type={isEditing ? 'text' : 'password'}
                    placeholder={t('llmConnectionForm.azure.apiKey.placeholder') || "Enter Azure OpenAI API key"}
                    error={errors.apiKey?.message}
                    readOnly={apiKeyReplaceMode}
                    showEndButton={apiKeyReplaceMode}
                    onEndButtonClick={() => {
                      setApiKeyReplaceMode(false);
                      setValue('apiKey', '');
                    }}
                    endButtonText={t('global.change') || "Change"}
                    {...field}
                  />
                )}
              />
            </div>
          </>
        );

      default:
        return null;
    }
  };

  const renderEmbeddingPlatformSpecificFields = () => {
    switch (selectedEmbeddingPlatform) {
      case 'aws':
        return (
          <>
            <div className="llm-connection-form__field"><p className="form-label">Embedding AWS region</p><Controller name="embeddingAwsRegion" control={control} rules={{required: 'Embedding AWS region is required'}} render={({field}) => <FormInput {...field} label="Embedding AWS region" placeholder="e.g. us-east-1" error={errors.embeddingAwsRegion?.message} readOnly={readOnly} />} /></div>
              <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.aws.embeddingAccessKey.label') || 'Embedding Access Key'}</p>
              <p className='form-description'>{t('llmConnectionForm.aws.embeddingAccessKey.description') || 'AWS Access Key for Bedrock embedding service'}</p>
              <Controller
                name="embeddingAccessKey"
                control={control}
                rules={{ 
                  required: t('llmConnectionForm.validationMessages.embeddingAccessKeyRequiredAws') || 'Embedding Access Key is required for AWS Bedrock',
                  pattern: {
                    value: /^\S+$/,
                    message: t('llmConnectionForm.validationMessages.invalidEmbeddingAccessKey') || 'Embedding Access Key cannot contain spaces'
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    type={isEditing ? 'text' : 'password'}
                    placeholder={t('llmConnectionForm.aws.embeddingAccessKey.placeholder') || "Enter AWS Access Key for embeddings"}
                    error={errors.embeddingAccessKey?.message}
                    readOnly={embeddingAccessKeyReplaceMode}
                    showEndButton={embeddingAccessKeyReplaceMode}
                    onEndButtonClick={() => {
                      setEmbeddingAccessKeyReplaceMode(false);
                      setValue('embeddingAccessKey', '');
                    }}
                    endButtonText={t('global.change') || "Change"}
                    {...field}
                  />
                )}
              />
            </div>
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.aws.embeddingSecretKey.label') || 'Embedding Secret Key'}</p>
              <p className='form-description'>{t('llmConnectionForm.aws.embeddingSecretKey.description') || 'AWS Secret Key for Bedrock embedding service'}</p>
              <Controller
                name="embeddingSecretKey"
                control={control}
                rules={{ 
                  required: t('llmConnectionForm.validationMessages.embeddingSecretKeyRequiredAws') || 'Embedding Secret Key is required for AWS Bedrock',
                  pattern: {
                    value: /^\S+$/,
                    message: t('llmConnectionForm.validationMessages.invalidEmbeddingSecretKey') || 'Embedding Secret Key cannot contain spaces'
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    type={isEditing ? 'text' : 'password'}
                    placeholder={t('llmConnectionForm.aws.embeddingSecretKey.placeholder') || "Enter AWS Secret Key for embeddings"}
                    error={errors.embeddingSecretKey?.message}
                    readOnly={embeddingSecretKeyReplaceMode}
                    showEndButton={embeddingSecretKeyReplaceMode}
                    onEndButtonClick={() => {
                      setEmbeddingSecretKeyReplaceMode(false);
                      setValue('embeddingSecretKey', '');
                    }}
                    endButtonText={t('global.change') || "Change"}
                    {...field}
                  />
                )}
              />
            </div>
          </>
        );
      case 'azure':
        return (
          <>
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.azure.embeddingDeploymentName.label') || 'Embedding Deployment Name'}</p>
              <p className='form-description'>{t('llmConnectionForm.azure.embeddingDeploymentName.description') || 'Azure OpenAI embedding deployment name'}</p>
              <Controller
                name="embeddingDeploymentName"
                control={control}
                rules={{ required: t('llmConnectionForm.validationMessages.embeddingDeploymentNameRequiredAzure') || 'Embedding Deployment Name is required for Azure OpenAI' }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    placeholder={t('llmConnectionForm.azure.embeddingDeploymentName.placeholder') || "Enter embedding deployment name"}
                    error={errors.embeddingDeploymentName?.message}
                    {...field}
                  />
                )}
              />
            </div>
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.azure.embeddingTargetUri.label') || 'Embedding Endpoint / Target URI'}</p>
              <p className='form-description'>{t('llmConnectionForm.azure.embeddingTargetUri.description') || 'Azure OpenAI embedding service endpoint URL'}</p>
              <Controller
                name="embeddingTargetUri"
                control={control}
                rules={{
                  required: t('llmConnectionForm.validationMessages.embeddingEndpointRequiredAzure') || 'Embedding Endpoint is required for Azure OpenAI',
                  pattern: {
                    value: /^https?:\/\/.+/,
                    message: t('llmConnectionForm.validationMessages.invalidUrl') || 'Please enter a valid URL starting with http:// or https://'
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    placeholder={t('llmConnectionForm.azure.targetUri.placeholder') || "https://your-resource.openai.azure.com/"}
                    error={errors.embeddingTargetUri?.message}
                    {...field}
                  />
                )}
              />
            </div>
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.azure.embeddingApiKey.label') || 'Embedding API Key'}</p>
              <p className='form-description'>{t('llmConnectionForm.azure.embeddingApiKey.description') || 'Azure OpenAI embedding API key'}</p>
              <Controller
                name="embeddingAzureApiKey"
                control={control}
                rules={{ 
                  required: t('llmConnectionForm.validationMessages.embeddingApiKeyRequiredAzure') || 'Embedding API Key is required for Azure OpenAI',
                  pattern: {
                    value: /^\S+$/,
                    message: t('llmConnectionForm.validationMessages.invalidEmbeddingApiKey') || 'Embedding API Key cannot contain spaces'
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    type={isEditing ? 'text' : 'password'}
                    placeholder={t('llmConnectionForm.azure.embeddingApiKey.placeholder') || "Enter Azure OpenAI embedding API key"}
                    error={errors.embeddingAzureApiKey?.message}
                    readOnly={embeddingAzureApiKeyReplaceMode}
                    showEndButton={embeddingAzureApiKeyReplaceMode}
                    onEndButtonClick={() => {
                      setEmbeddingAzureApiKeyReplaceMode(false);
                      setValue('embeddingAzureApiKey', '');
                    }}
                    endButtonText={t('global.change') || "Change"}
                    {...field}
                  />
                )}
              />
            </div>
          </>
        );

      default:
        return null;
    }
  };

  const handleFormSubmit = (data: LLMConnectionFormData) => {
    const cleanedData = {
      ...data,
      monthlyBudget: data.monthlyBudget.replace(/,/g, ''),
      warnBudget: data.warnBudget.replace('%', ''),
      stopBudget: data.stopBudget.replace('%', ''),
    };
    onSubmit(cleanedData);
  };

  return (
    <div className="llm-connection-form">
      <form onSubmit={handleSubmit(handleFormSubmit)}>
        <div className="form-section">
          <h3 className="form-section-title">{t('llmConnectionForm.sections.llmConfiguration') || 'LLM Configuration'}</h3>

          <div className="form-row">
            <p className='form-label'>{t('llmConnectionForm.fields.connectionName.label') || 'Connection Name'}</p>
            <p className='form-description'>{t('llmConnectionForm.fields.connectionName.description') || 'A unique name to identify this LLM connection'}</p>
            <Controller
              name="connectionName"
              control={control}
              rules={{ 
                required: t('llmConnectionForm.validationMessages.connectionNameRequired') || 'Connection Name is required',
                maxLength: {
                  value: 100,
                  message: t('llmConnectionForm.validationMessages.connectionNameMaxLength') || 'Connection Name must not exceed 100 characters'
                }
              }}
              render={({ field }) => (
                <FormInput
                  label=""
                  placeholder={t('llmConnectionForm.fields.connectionName.placeholder') || "Enter connection name (e.g., Azure GPT-4 Production)"}
                  error={errors.connectionName?.message}
                  disabled={readOnly}
                  //maxLength={100}
                  {...field}
                />
              )}
            />
          </div>

          <div className="form-row">
            <p className='form-label'>{t('llmConnectionForm.fields.llmPlatform.label') || 'LLM Platform'}</p>
            <p className='form-description'>{t('llmConnectionForm.fields.llmPlatform.description') || 'Cloud / local platform in which your model is hosted'}</p>
            <Controller
              name="llmPlatform"
              control={control}
              rules={{ required: t('llmConnectionForm.validationMessages.llmPlatformRequired') || 'LLM Platform is required' }}
              render={({ field }) => (
                <FormSelect
                  label=""
                  options={llmPlatformOptions || []}
                  placeholder={
                    llmPlatformsLoading
                      ? t('llmConnectionForm.placeholders.loadingPlatforms') || "Loading platforms..."
                      : llmPlatformsError
                        ? t('llmConnectionForm.placeholders.errorLoadingPlatforms') || "Error loading platforms"
                        : t('llmConnectionForm.fields.llmPlatform.placeholder') || "Select LLM Platform"
                  }
                  error={errors.llmPlatform?.message || (llmPlatformsError ? t('llmConnectionForm.validationMessages.failedToLoadPlatforms') || "Failed to load platforms" : undefined)}
                  disabled={readOnly || llmPlatformsLoading}
                  onSelectionChange={(selected) => {
                    field.onChange(selected?.value || '');
                    resetLLMCredentialFields();
                  }}
                  defaultValue={field.value}
                  {...field}
                />
              )}
            />
          </div>

          <div className="form-row">
            <p className='form-label'>{t('llmConnectionForm.fields.llmModel.label') || 'LLM Model'}</p>
            <p className='form-description'>{t('llmConnectionForm.fields.llmModel.description') || 'The LLM model that you want to use'}</p>

            <Controller
              name="llmModel"
              control={control}
              rules={{ required: t('llmConnectionForm.validationMessages.llmModelRequired') || 'LLM Model is required' }}
              render={({ field }) => (
                <FormSelect
                  label=""
                  options={getLLMModelOptions() || []}
                  placeholder={
                    llmModelsLoading
                      ? t('llmConnectionForm.placeholders.loadingModels') || "Loading models..."
                      : llmModelsError
                        ? t('llmConnectionForm.placeholders.errorLoadingModels') || "Error loading models"
                        : !selectedLLMPlatform
                          ? t('llmConnectionForm.placeholders.selectPlatformFirst') || "Select a platform first"
                          : t('llmConnectionForm.fields.llmModel.placeholder') || "Select LLM Model"
                  }
                  error={errors.llmModel?.message || (llmModelsError ? t('llmConnectionForm.validationMessages.failedToLoadModels') || "Failed to load models" : undefined)}
                  disabled={!selectedLLMPlatform || readOnly || llmModelsLoading}
                  onSelectionChange={(selected) => {
                    field.onChange(selected?.value || '');
                  }}
                  defaultValue={field.value}
                  {...field}
                />
              )}
            />
          </div>

          {/* Platform-specific fields */}
          {renderPlatformSpecificFields()}
        </div>

        <div className="form-section">
          <h3 className="form-section-title">{t('llmConnectionForm.sections.embeddingConfiguration') || 'Embedding Model Configuration'}</h3>

          <div className="form-row">
            <p className='form-label'>{t('llmConnectionForm.fields.embeddingPlatform.label') || 'Embedding Model Platform'}</p>
            <p className='form-description'>{t('llmConnectionForm.fields.embeddingPlatform.description') || 'This is the cloud / local platform in which your embedding model is hosted'}</p>

            <Controller
              name="embeddingModelPlatform"
              control={control}
              rules={{ required: t('llmConnectionForm.validationMessages.embeddingPlatformRequired') || 'Embedding Model Platform is required' }}
              render={({ field }) => (
                <FormSelect
                  label=""
                  options={embeddingPlatformOptions || []}
                  placeholder={
                    embeddingPlatformsLoading
                      ? t('llmConnectionForm.placeholders.loadingPlatforms') || "Loading platforms..."
                      : embeddingPlatformsError
                        ? t('llmConnectionForm.placeholders.errorLoadingPlatforms') || "Error loading platforms"
                        : t('llmConnectionForm.fields.embeddingPlatform.placeholder') || "Select Embedding Platform"
                  }
                  error={errors.embeddingModelPlatform?.message || (embeddingPlatformsError ? t('llmConnectionForm.validationMessages.failedToLoadPlatforms') || "Failed to load platforms" : undefined)}
                  disabled={embeddingPlatformsLoading}
                  onSelectionChange={(selected) => {
                    field.onChange(selected?.value || '');
                    resetEmbeddingModelCredentialFields();
                  }}
                  defaultValue={field.value}
                  {...field}
                />
              )}
            />
          </div>

          <div className="form-row">
            <p className='form-label'>{t('llmConnectionForm.fields.embeddingModel.label') || 'Embedding Model'}</p>
            <p className='form-description'>{t('llmConnectionForm.fields.embeddingModel.description') || 'The embedding model that will be used for searching your knowledge base'}</p>

            <Controller
              name="embeddingModel"
              control={control}
              rules={{ required: t('llmConnectionForm.validationMessages.embeddingModelRequired') || 'Embedding Model is required' }}
              render={({ field }) => (
                <FormSelect
                  label=""
                  options={getEmbeddingModelOptions() || []}
                  placeholder={
                    embeddingModelsLoading
                      ? t('llmConnectionForm.placeholders.loadingModels') || "Loading models..."
                      : embeddingModelsError
                        ? t('llmConnectionForm.placeholders.errorLoadingModels') || "Error loading models"
                        : !selectedEmbeddingPlatform
                          ? t('llmConnectionForm.placeholders.selectPlatformFirst') || "Select a platform first"
                          : t('llmConnectionForm.fields.embeddingModel.placeholder') || "Select Embedding Model"
                  }
                  error={errors.embeddingModel?.message || (embeddingModelsError ? t('llmConnectionForm.validationMessages.failedToLoadModels') || "Failed to load models" : undefined)}
                  disabled={!selectedEmbeddingPlatform || embeddingModelsLoading}
                  onSelectionChange={(selected) => {
                    field.onChange(selected?.value || '');
                  }}
                  defaultValue={field.value}
                  {...field}
                />
              )}
            />
          </div>

          {/* Embedding Platform-specific fields */}
          {renderEmbeddingPlatformSpecificFields()}
        </div>

        <div className="form-section">
          <h3 className="form-section-title">{t('llmConnectionForm.sections.budgetDeployment') || 'Budget & Deployment'}</h3>

          <div className="form-row">
            <p className='form-label'>{t('llmConnectionForm.fields.monthlyBudget.label') || 'Monthly Budget'}</p>
            <p className='form-description'>{t('llmConnectionForm.fields.monthlyBudget.description') || 'Total monthly budget including embedding model and LLM model. If the LLM integration usage cost exceeds the below budget, the LLM will respond with an "inactive" status'}</p>

            <Controller
              name="monthlyBudget"
              control={control}
              rules={{
                required: t('llmConnectionForm.validationMessages.monthlyBudgetRequired') || 'Monthly Budget is required',
                pattern: {
                  value: /^[\d,]+(\.\d{1,2})?$/,
                  message: t('llmConnectionForm.validationMessages.monthlyBudgetInvalid') || 'Please enter a valid budget amount'
                },
                validate: value => {
                  const numericValue = value.replace(/,/g, '');
                  return Number(numericValue) > 0 || t('llmConnectionForm.validationMessages.monthlyBudgetPositive') || 'Monthly Budget must be a positive number';
                }
              }}
              render={({ field }) => (
                <FormInput
                  label=""
                  placeholder={t('llmConnectionForm.fields.monthlyBudget.placeholder') || "Enter monthly budget"}
                  error={errors.monthlyBudget?.message}
                  {...field}
                  prefix='€'
                  formatAsNumber={true}
                />
              )}
            />
          </div>

          <div className="form-row">

            <Controller
              name="disconnectOnBudgetExceed"
              control={control}
              render={({ field }) => (
                <FormCheckbox
                  label=""
                  name="disconnectOnBudgetExceed"
                  item={{
                    label: t('llmConnectionForm.fields.disconnectOnBudgetExceed.label') || "Automatically disconnect LLM connection when stop budget threshold is exceeded",
                    value: "true",
                    checked: field.value
                  }}
                  checked={field.value}
                  onChange={(e) => field.onChange(e.target.checked)}
                  hideLabel={true}
                />
              )}
            />
          </div>

          <div className="form-row">
            <p className='form-label'>{t('llmConnectionForm.fields.warnBudget.label') || 'Warn Budget Threshold'}</p>
            <p className='form-description'>{t('llmConnectionForm.fields.warnBudget.description') || 'You will get a notification when your usage reaches this percentage of your allocated monthly budget.'}</p>

            <Controller
              name="warnBudget"
              control={control}
              rules={{
                required: t('llmConnectionForm.validationMessages.warnBudgetRequired') || 'Warn Budget Threshold is required',
                pattern: {
                  value: /^\d+$/,
                  message: t('llmConnectionForm.validationMessages.numbersOnly') || 'Please enter numbers only'
                },
                validate: (value, formValues) => {
                  const numericValue = Number(value.replace('%', ''));

                  if (numericValue < 1 || numericValue > 100) {
                    return t('llmConnectionForm.validationMessages.warnBudgetRange') || 'Warn Budget Threshold must be between 1-100%';
                  }
                  return true;
                }
              }}
              render={({ field }) => (
                <FormInput
                  label=""
                  placeholder={t('llmConnectionForm.fields.warnBudget.placeholder') || "Enter warn budget threshold"}
                  error={errors.warnBudget?.message}
                  value={field.value ? `${field.value}%` : ''}
                  onChange={(e) => {
                    const value = e.target.value.replace(/[^\d]/g, ''); // Remove all non-numeric characters
                    field.onChange(value);
                  }}
                  name={field.name}
                  onBlur={field.onBlur}
                />
              )}
            />
          </div>

          {disconnectOnBudgetExceed && (
            <div className="form-row">
              <p className='form-label'>{t('llmConnectionForm.fields.stopBudget.label') || 'Disconnect Budget Threshold'}</p>
              <p className='form-description'>{t('llmConnectionForm.fields.stopBudget.description') || 'Your LLM connection will be automatically disconnected and all further requests will be stopped when your usage reaches this percentage of your monthly budget.'}</p>

              <Controller
                name="stopBudget"
                control={control}
                rules={{
                  required: disconnectOnBudgetExceed ? t('llmConnectionForm.validationMessages.stopBudgetRequired') || 'Stop Budget Threshold is required' : false,
                  pattern: {
                    value: /^\d+$/,
                    message: t('llmConnectionForm.validationMessages.numbersOnly') || 'Please enter numbers only'
                  },
                  validate: (value, formValues) => {
                    if (!disconnectOnBudgetExceed) return true;

                    const numericValue = Number(value.replace('%', ''));
                    const warnValue = Number(formValues.warnBudget?.replace('%', '') || 0);

                    if (numericValue < 1 || numericValue > 200) {
                      return t('llmConnectionForm.validationMessages.stopBudgetRange') || 'Stop Budget Threshold must be between 1-200%';
                    }

                    if (warnValue > 0 && numericValue <= warnValue) {
                      return t('llmConnectionForm.validationMessages.stopBudgetGreater') || 'Stop Budget Threshold must be greater than Warn Budget Threshold';
                    }

                    return true;
                  }
                }}
                render={({ field }) => (
                  <FormInput
                    label=""
                    placeholder={t('llmConnectionForm.fields.stopBudget.placeholder') || "Enter stop budget threshold"}
                    error={errors.stopBudget?.message}
                    value={field.value ? `${field.value}%` : ''}
                    onChange={(e) => {
                      const value = e.target.value.replace(/[^\d]/g, ''); // Remove all non-numeric characters
                      field.onChange(value);
                    }}
                    name={field.name}
                    onBlur={field.onBlur}
                  />
                )}
              />
            </div>
          )}

          <div className="form-row">
            <Controller
              name="deploymentEnvironment"
              control={control}
              rules={{ required: t('llmConnectionForm.validationMessages.deploymentEnvironmentRequired') || 'Deployment Environment is required' }}
              render={({ field }) => (
                <div className="radio-group">
                  <label className="radio-group-label">{t('llmConnectionForm.fields.deploymentEnvironment.label') || 'Deployment Environment'}</label>
                  <div className="radio-options">
                    {deploymentEnvironments?.map((env) => (
                      <label key={env.value} className="radio-option">
                        <input
                          type="radio"
                          value={env.value}
                          checked={field.value === env.value}
                          onChange={(e) => field.onChange(e.target.value)}
                          className="radio-input"
                        />
                        <span className="radio-label">{env.label}</span>
                      </label>
                    ))}
                  </div>
                  {errors.deploymentEnvironment && (
                    <p className="input__inline_error">{errors.deploymentEnvironment.message}</p>
                  )}
                </div>
              )}
            />
          </div>
          <Track className="form-footer" gap={16} justify="end">
            <div className="flex-grid">
              <Button
                appearance="secondary"
                onClick={onCancel}
                type="button"
              >
                {t('global.cancel') || 'Cancel'}
              </Button>
              {isEditing && (<Button
                appearance="error"
                onClick={onDelete}
                type='button'
              >
                {t('llmConnectionForm.buttons.deleteConnection') || 'Delete Connection'}
              </Button>)}
              <Button
                type="submit"
                disabled={!isDirty || !isValid}
                appearance="primary"
              >
                {isEditing ? (t('llmConnectionForm.buttons.updateConnection') || 'Update Connection') : (t('llmConnectionForm.buttons.createConnection') || 'Create Connection')}
              </Button>
            </div>
          </Track>
        </div>

      </form>
    </div>
  );
};

export default LLMConnectionForm;
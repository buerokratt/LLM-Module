import { FC, PropsWithChildren, useState } from 'react';
import Button from 'components/Button';
import Label from 'components/Label';
import { useDialog } from 'hooks/useDialog';
import './LLMConnectionCard.scss';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Switch } from 'components/FormElements';
import {
  updateLLMConnectionStatus,
  EmbeddingModelLockedError,
  EmbeddingCheckUnavailableError,
} from 'services/llmConnections';
import { embeddingLockMessage } from 'utils/embeddingLockMessage';
import { useToast } from 'hooks/useToast';
import { ToastTypes } from 'enums/commonEnums';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { llmConnectionsQueryKeys } from 'utils/queryKeys';
import { AxiosError } from 'axios';

type LLMConnectionCardProps = {
  llmConnectionId: number | string;
  llmConnectionName?: string;
  platform?: string;
  model?: string;
  isActive?: boolean;
  deploymentEnv?: string;
  budgetStatus?: string;
  usedBudget?: number;
  monthlyBudget?: number;
  stopBudgetThreshold?: number;
  disconnectOnBudgetExceed?: boolean;
  onStatusChange?: (id: number | string, newStatus: boolean) => void;
};

const LLMConnectionCard: FC<PropsWithChildren<LLMConnectionCardProps>> = ({
  llmConnectionId,
  llmConnectionName,
  platform,
  model,
  isActive,
  deploymentEnv,
  budgetStatus,
  usedBudget,
  monthlyBudget,
  stopBudgetThreshold,
  disconnectOnBudgetExceed,
  onStatusChange,
}) => {
  const { open, close } = useDialog();
  const { t } = useTranslation();
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();

  // Format currency
  const formatCurrency = (amount?: number): string => {
  if (amount === undefined || amount === null) {
    return '0,00 €';
  }

  return new Intl.NumberFormat('et-EE', {
    style: 'currency',
    currency: 'EUR',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
 };


  // Get the relevant budget threshold
  const getRelevantBudget = (): number | undefined => {
    // here if disconnect on budget exceed is enabled and stop threshold is set, calculate the actual amount from percentage
    if (disconnectOnBudgetExceed && stopBudgetThreshold && stopBudgetThreshold > 0 && monthlyBudget) {
      return (monthlyBudget * stopBudgetThreshold) / 100;
    }
    // Otherwise using monthly budget
    return monthlyBudget;
  };

  const updateStatusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string | number; status: 'active' | 'inactive' }) =>
      updateLLMConnectionStatus(id, status),
    onSuccess: async (data, variables) => {
      // Invalidate queries to refresh the data
      await queryClient.invalidateQueries({
        queryKey: llmConnectionsQueryKeys.all()
      });
      
      toast.open({
        type: ToastTypes.SUCCESS,
        title: t('toast.success.title'),
        message: `Connection ${variables.status === 'active' ? 'activated' : 'deactivated'} successfully`,
      });
      
      // Call the parent callback to update the list immediately
      if (onStatusChange) {
        onStatusChange(llmConnectionId, variables.status === 'active');
      }
    },
    onError: (error: AxiosError | EmbeddingModelLockedError | EmbeddingCheckUnavailableError) => {
      console.error('Error updating connection status:', error);
      // Activation refused by the embedding model lock (#806): one short, specific message
      const message =
        embeddingLockMessage(t, error, 'notActivated') || 'Failed to update connection status';
      toast.open({
        type: ToastTypes.ERROR,
        title: t('toast.error.title'),
        message,
      });
    },
  });

  const handleStatusChange = async (checked: boolean) => {
    if (updateStatusMutation.isLoading) return;
    
    const newStatus = checked ? 'active' : 'inactive';
    updateStatusMutation.mutate({
      id: llmConnectionId,
      status: newStatus
    });
  };


  const renderDeploymentEnv = (deploymentEnvironment: string | undefined) => {
    if (deploymentEnvironment === "testing") {
      return (
        <Label type="info">
          {t('dataModels.environments.testing')}
        </Label>
      );
    } else if (deploymentEnvironment === "production") {
      return (
        <Label type="success">
          {t('dataModels.environments.production')}
        </Label>
      );
    }
  };

  const renderBudgetStatus = (status: string | undefined) => {
    if (status === "within_budget") {
      return (
        <Label type="success">
          {t('dataModels.budgetStatus.withinBudget')}
        </Label>
      );
    } else if (status === "over_budget") {
      return (
        <Label type="error">
          {t('dataModels.budgetStatus.overBudget')}
        </Label>
      );
    } else if (status === "close_to_exceed") {
      return (
        <Label type="warning">
          {t('dataModels.budgetStatus.closeToExceed')}
        </Label>
      );
    }
  };

  return (
    <div>
      <div className="dataset-group-card">
        <div className="flex space-between">
          <p>{llmConnectionName}</p>
          <Switch
            label=""
            checked={isActive ?? false}
            onCheckedChange={handleStatusChange}
            disabled={updateStatusMutation.isLoading}
          />
        </div>

        <div className="flex" style={{ flexWrap: 'wrap', gap: '5px' }}>
          <div className="label-row">
            <span className="label-title">
              {t('dataModels.filters.platform')}:
            </span>
            <span className="label-value">{platform ?? 'N/A'}</span>
          </div>
          <div className="label-row">
            <span className="label-title">
              {t('dataModels.filters.model')}:
            </span>
            <span className="label-value">{model ?? 'N/A'}</span>
          </div>
          {(usedBudget !== undefined || monthlyBudget !== undefined) && (
            <div className="label-row">
              <span className="label-title">
                {t('dataModels.budgetUsage')}:
              </span>
              <span className="label-value">
                {formatCurrency(usedBudget)} / {formatCurrency(getRelevantBudget())}
              </span>
            </div>
          )}
          <div className='label-row'>
            {renderDeploymentEnv(deploymentEnv)}
            {renderBudgetStatus(budgetStatus)}
          </div>
        </div>
        <div className="button-row mt-3">
          <Button
            appearance="secondary"
            size="s"
            onClick={() => navigate(`/view-llm-connection?id=${llmConnectionId}`)}
          >
            {t('dataModels.settings') ?? ''}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default LLMConnectionCard;
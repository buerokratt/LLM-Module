import BackArrowButton from "assets/BackArrowButton";
import LLMConnectionForm, { LLMConnectionFormData } from "components/molecules/LLMConnectionForm";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { useDialog } from 'hooks/useDialog';
import { createLLMConnection, CreateLLMConnectionError, getProductionConnection } from 'services/llmConnections';
import { llmConnectionsQueryKeys } from 'utils/queryKeys';
import { ButtonAppearanceTypes } from 'enums/commonEnums';
import { Button } from 'components';

const CreateLLMConnection = () => {
    const { t } = useTranslation();
    const navigate = useNavigate();
    const { open: openDialog, close: closeDialog } = useDialog();
    const queryClient = useQueryClient();
    
    // Query to check for existing production connection
    const { data: existingProductionConnection } = useQuery({
      queryKey: llmConnectionsQueryKeys.production(),
      queryFn: () => getProductionConnection(),
    });
    
    const createConnectionMutation = useMutation({
      mutationFn: createLLMConnection,
      onSuccess: async () => {
        await queryClient.invalidateQueries({
          queryKey: llmConnectionsQueryKeys.all()
        });
        
        openDialog({
          title: t('llmConnectionForm.createConnection.successDialogTitle') || 'Connection Succeeded',
          content: <p>{t('llmConnectionForm.createConnection.successDialogMessage') || 'The provided LLM configuration is successfully configured'}</p>,
          footer: (
            <Button
              appearance={ButtonAppearanceTypes.PRIMARY}
              onClick={() => {
                closeDialog();
                navigate('/llm-connections');
              }}
            >
              {t('llmConnectionForm.createConnection.viewConnectionsButton') || 'View LLM Connections'}
            </Button>
          ),
        });
      },
      onError: async (error: unknown) => {
        console.error('Error creating LLM connection:', error);

        // A failed create can still have changed connections (e.g. created as testing), so refresh the lists
        await queryClient.invalidateQueries({
          queryKey: llmConnectionsQueryKeys.all()
        });

        const stage = error instanceof CreateLLMConnectionError ? error.stage : 'save';
        const noProductionConnection = error instanceof CreateLLMConnectionError && error.noProductionConnection;

        openDialog({
          title: stage === 'promote'
            ? t('llmConnectionForm.createConnection.errors.promoteTitle')
            : t('llmConnectionForm.createConnection.errorDialogTitle') || 'Connection Failed',
          content: (
            <div>
              <p>{t(`llmConnectionForm.createConnection.errors.${stage}`)}</p>
              {noProductionConnection && (
                <p>{t('llmConnectionForm.createConnection.errors.noProductionConnection')}</p>
              )}
            </div>
          ),
          footer: (
            <Button
              appearance={ButtonAppearanceTypes.PRIMARY}
              onClick={() => {
                closeDialog();
                // The connection exists as testing, so show it in the list
                if (stage === 'promote') {
                  navigate('/llm-connections');
                }
              }}
            >
              {stage === 'promote'
                ? t('llmConnectionForm.createConnection.viewConnectionsButton') || 'View LLM Connections'
                : t('llmConnectionForm.createConnection.goBackButton') || 'Go Back'}
            </Button>
          ),
        });
      },
    });
    
    const handleSubmit = async (data: LLMConnectionFormData) => {
      const isCreatingProductionConnection = data.deploymentEnvironment === 'production';
      const hasExistingProductionConnection = existingProductionConnection && existingProductionConnection.id;
      
      if (isCreatingProductionConnection && hasExistingProductionConnection) {
        openDialog({
          title: t('llmConnectionForm.createConnection.replaceProductionDialogTitle') || 'Replace Production Connection',
          content: (
            <div>
              <p>{t('llmConnectionForm.createConnection.replaceProductionDialogMessage', { connectionName: existingProductionConnection.connectionName }) || `A production connection "${existingProductionConnection.connectionName}" already exists.`}</p>
              <p>{t('llmConnectionForm.createConnection.replaceProductionDialogWarning') || 'Creating this new production connection will replace the current one. Are you sure you want to proceed?'}</p>
            </div>
          ),
          footer: (
            <div className="button-wrapper">
              <Button
                appearance={ButtonAppearanceTypes.SECONDARY}
                onClick={closeDialog}
              >
                {t('llmConnectionForm.createConnection.cancelButton') || 'Cancel'}
              </Button>
              <Button
                appearance={ButtonAppearanceTypes.PRIMARY}
                onClick={() => {
                  createConnectionMutation.mutate(data);
                }}
                showLoadingIcon={createConnectionMutation.isLoading}
              >
                {t('llmConnectionForm.createConnection.confirmReplaceButton') || 'Yes, Replace Production Connection'}
              </Button>
            </div>
          ),
        });
      } else {
        createConnectionMutation.mutate(data);
      }
    };
    
    const handleCancel = () => {
      navigate('/llm-connections');
    };

    return(
        <div className="container">
        <div className="title_container">
          <div className="flex-grid">
            <Link to={'/llm-connections'}>
              <BackArrowButton />
            </Link>
            <div className="title">{t('llmConnectionForm.createConnection.title') || 'Create LLM Connection'}</div>
          </div>
        </div>
        <LLMConnectionForm
        onSubmit={handleSubmit}
        onCancel={handleCancel}
        isEditing={false}
        onDelete={() => {}}
      />
        </div>

    )
}

export default CreateLLMConnection;
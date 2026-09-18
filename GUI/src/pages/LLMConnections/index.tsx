import { FC, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, FormSelect } from 'components';
import Pagination from 'components/molecules/Pagination';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { formattedArray } from 'utils/commonUtils';
import DataModelCard from 'components/molecules/LLMConnectionCard';
import CircularSpinner from 'components/molecules/CircularSpinner/CircularSpinner';
import { ButtonAppearanceTypes } from 'enums/commonEnums';
import NoDataView from 'components/molecules/NoDataView';
import BudgetBanner from 'components/molecules/BudgetBanner';
import './LLMConnections.scss';
import { platforms, trainingStatuses } from 'config/dataModelsConfig';
import LLMConnectionCard from 'components/molecules/LLMConnectionCard';
import { fetchLLMConnectionsPaginated, LLMConnectionFilters, LLMConnection, getProductionConnection, ProductionConnectionFilters } from 'services/llmConnections';
import { llmConnectionsQueryKeys } from 'utils/queryKeys';
import { useToast } from 'hooks/useToast';
import { ToastTypes } from 'enums/commonEnums';
import useStore from 'store';
import { getAllLLMModels, getLLMPlatforms } from 'services/llmConfigs';

const LLMConnections: FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const toast = useToast();

  // Use Zustand store for persistent filters
  const {
    llmConnectionFilters: filters,
    llmConnectionPageIndex: pageIndex,
    productionConnectionFilters,
    setLLMConnectionFilters: setFilters,
    setLLMConnectionPageIndex: setPageIndex,
    setProductionConnectionFilters,
    resetLLMConnectionFilters,
  } = useStore();

  // Fetch LLM connections using TanStack Query with new paginated endpoint
  const { data: connectionsResponse, isLoading: isModelDataLoading, error } = useQuery({
    queryKey: llmConnectionsQueryKeys.paginatedList(filters),
    queryFn: () => fetchLLMConnectionsPaginated(filters),
  });

  // Fetch production connection separately with filters from store
  const { data: productionConnection, isLoading: isProductionLoading } = useQuery({
    queryKey: llmConnectionsQueryKeys.production(productionConnectionFilters),
    queryFn: () => getProductionConnection(productionConnectionFilters),
  });


    // Fetch platform and model options from API
  const { data: llmPlatformsData = [], isLoading: llmPlatformsLoading, error: llmPlatformsError } = useQuery({
      queryKey: ['llm-platforms'],
      queryFn: getLLMPlatforms
    });

  const { data: llmModels= [], isLoading: llmModelsLoading, error: llmModelsError } = useQuery({
      queryKey: ['llm-models'],
      queryFn: getAllLLMModels,
    });
  
  const llmConnections = connectionsResponse;
  const totalPages = connectionsResponse?.[0]?.totalPages || 1;

  // Update filters when pageIndex changes
  useEffect(() => {
    setFilters({ ...filters, pageNumber: pageIndex });
  }, [pageIndex, setFilters]);

  // Sync production filters with main filters
  useEffect(() => {
    setProductionConnectionFilters({
      llmPlatform: filters.llmPlatform || '',
      llmModel: filters.llmModel || '',
      sortBy: filters.sortBy || 'created_at',
      sortOrder: filters.sortOrder || 'desc',
    });
  }, [filters.llmPlatform, filters.llmModel, filters.sortBy, filters.sortOrder, setProductionConnectionFilters]);

  // Show toast on error
  useEffect(() => {
    if (error) {
      toast.open({
        type: ToastTypes.ERROR,
        title: t('toast.error.title') || 'Error',
        message: t('dataModels.errorLoadingConnections') || 'Error loading LLM connections',
      });
    }
  }, [error, toast, t]);

  const handleFilterChange = (
    name: string,
    value: string | number | undefined | { name: string; id: string }
  ) => {
    let filterUpdate: Partial<LLMConnectionFilters> = {};

    if (name === 'sorting') {
      // Handle sorting format - no conversion needed, use snake_case directly
      const sortingValue = value as string;
      const [sortBy, sortOrder] = sortingValue.split(' ');

      filterUpdate = {
        sortBy: sortBy,
        sortOrder: sortOrder as 'asc' | 'desc' 
      };
    } else {
      filterUpdate = { [name]: value };
    }

    setFilters({
      ...filters,
      ...filterUpdate,
    });

    // Reset to first page when filters change
    if (name !== 'pageNumber') {
      setPageIndex(1);
    }
  };
  
  const platformOptions = [
    { label: t('dataModels.filters.allPlatforms'), value: 'all' },
    ...llmPlatformsData.map((platform) => ({
      label: platform.label,
      value: platform.value,
    })),
  ];

  const llmModelOptions = [
    { label: t('dataModels.filters.allModels'), value: 'all' },
    ...llmModels.map((model) => ({
      label: model.label,
      value: model.value,
    })),
  ];

  // Environment filter options
  const environmentOptions = [
    { label: t('dataModels.filters.allEnvironments'), value: 'all' },
    { label: t('dataModels.environments.testing'), value: 'testing' },
    { label: t('dataModels.environments.production'), value: 'production' },
  ];

  // Sort options - using snake_case format for backend
  const sortOptions = [
    { label: t('dataModels.sortOptions.createdDateNewest'), value: 'created_at desc' },
    { label: t('dataModels.sortOptions.createdDateOldest'), value: 'created_at asc' },
    { label: t('dataModels.sortOptions.platformAZ'), value: 'llm_platform asc' },
    { label: t('dataModels.sortOptions.platformZA'), value: 'llm_platform desc' },
    { label: t('dataModels.sortOptions.modelAZ'), value: 'llm_model asc' },
    { label: t('dataModels.sortOptions.modelZA'), value: 'llm_model desc' },
    { label: t('dataModels.sortOptions.budgetHighToLow'), value: 'monthly_budget desc' },
    { label: t('dataModels.sortOptions.budgetLowToHigh'), value: 'monthly_budget asc' },
  ];

  const currentSorting = `${filters.sortBy || 'created_at'} ${filters.sortOrder || 'desc'}`;

  // Use production connection as featured connection
  const otherConnections = llmConnections || [];

  return (
    <div>
      <div className="container">
        {!isModelDataLoading && !isProductionLoading ? (
          <div>
            <div>
              <div className="title_container">
                <div className="title">{t('dataModels.dataModels')}</div>
                <Button
                  appearance="primary"
                  size="m"
                  onClick={() => navigate('/create-llm-connection')}
                >
                  {t('dataModels.createModel')}
                </Button>
              </div>
              <div className="search-panel">
                <div className="models-filter-div">
                  <FormSelect
                    label=""
                    name="llmPlatform"
                    placeholder={t('dataModels.filters.platform') ?? 'Platform'}
                    options={platformOptions}
                    onSelectionChange={(selection) =>
                      handleFilterChange('llmPlatform', selection?.value === 'all' ? '' : selection?.value)
                    }
                    defaultValue={filters?.llmPlatform || 'all'}
                  />
                  <FormSelect
                    label=""
                    name="llmModel"
                    placeholder={t('dataModels.filters.model') ?? 'Model'}
                    options={llmModelOptions}
                    onSelectionChange={(selection) =>
                      handleFilterChange('llmModel', selection?.value === 'all' ? '' : selection?.value)
                    }
                    defaultValue={filters?.llmModel || 'all'}
                  />
                  <FormSelect
                    label=""
                    name="environment"
                    placeholder={t('dataModels.filters.environment') ?? 'Environment'}
                    options={environmentOptions}
                    onSelectionChange={(selection) =>
                      handleFilterChange('environment', selection?.value === 'all' ? '' : selection?.value)
                    }
                    defaultValue={filters?.environment || 'all'}
                  />

                  <FormSelect
                    label=""
                    name="sorting"
                    placeholder={t('dataModels.filters.sortBy') ?? 'Sort By'}
                    options={sortOptions}
                    onSelectionChange={(selection) =>
                      handleFilterChange('sorting', selection?.value)
                    }
                    defaultValue={currentSorting}
                  />

                  <div className="filter-reset-button">
                    <Button
                      onClick={resetLLMConnectionFilters}
                      appearance={ButtonAppearanceTypes.SECONDARY}
                    >
                      {t('global.reset') ?? 'Reset'}
                    </Button>
                  </div>
                </div>
              </div>

              {productionConnection && filters?.environment !== "testing" && (
                <div className="m-30-0">
                  <p>{t('dataModels.productionConnections')}</p>
                  <div className="grid-container m-30-0">
                    <LLMConnectionCard
                      key={productionConnection.id}
                      llmConnectionId={productionConnection.id}
                      llmConnectionName={productionConnection.connectionName}
                      isActive={productionConnection.connectionStatus === 'active'}
                      deploymentEnv={productionConnection.environment}
                      budgetStatus={productionConnection.budgetStatus}
                      platform={productionConnection.llmPlatform}
                      model={productionConnection.llmModel}                     
                      usedBudget={productionConnection.usedBudget}
                      monthlyBudget={productionConnection.monthlyBudget}
                      stopBudgetThreshold={productionConnection.stopBudgetThreshold}
                      disconnectOnBudgetExceed={productionConnection.disconnectOnBudgetExceed}
                    />
                  </div>
                </div>
              )}

              {otherConnections?.length > 0 ? (
                <div>
                  <p>{t('dataModels.otherConnections')}</p>
                  <div className="grid-container m-30-0">
                    {otherConnections?.map((llmConnection: LLMConnection) => {
                      return (
                        <LLMConnectionCard
                          key={llmConnection.id}
                          llmConnectionId={llmConnection.id}
                          llmConnectionName={llmConnection.connectionName}
                          isActive={llmConnection.connectionStatus === 'active'}
                          deploymentEnv={llmConnection.environment}
                          budgetStatus={llmConnection.budgetStatus}
                          platform={llmConnection.llmPlatform}
                          model={llmConnection.llmModel}
                          usedBudget={llmConnection.usedBudget}
                          monthlyBudget={llmConnection.monthlyBudget}
                          stopBudgetThreshold={llmConnection.stopBudgetThreshold}
                          disconnectOnBudgetExceed={llmConnection.disconnectOnBudgetExceed}
                        />
                      );
                    })}
                  </div>
                </div>
              ) : !productionConnection ? (
                <NoDataView text={t('dataModels.noModels') ?? 'No LLM connections found'} />
              ) : null}
              
            </div>
            <Pagination
              pageCount={totalPages}
              pageIndex={pageIndex}
              canPreviousPage={pageIndex > 1}
              canNextPage={pageIndex < totalPages}
              onPageChange={setPageIndex}
            />
          </div>
        ) : (
          <CircularSpinner />
        )}
      </div>
    </div>
  );
};

export default LLMConnections;
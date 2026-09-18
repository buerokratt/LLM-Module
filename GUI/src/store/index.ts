import { create } from 'zustand';
import { UserInfo } from 'types/userInfo';
import { LLMConnectionFilters, ProductionConnectionFilters } from 'services/llmConnections';

interface StoreState {
  userInfo: UserInfo | null;
  userId: string;
  userDomains: string[];
  setUserInfo: (info: UserInfo) => void;
  setUserDomains: (domains: string[]) => void;
  llmConnectionFilters: LLMConnectionFilters;
  llmConnectionPageIndex: number;
  productionConnectionFilters: ProductionConnectionFilters;
  setLLMConnectionFilters: (filters: LLMConnectionFilters) => void;
  setLLMConnectionPageIndex: (pageIndex: number) => void;
  setProductionConnectionFilters: (filters: ProductionConnectionFilters) => void;
  resetLLMConnectionFilters: () => void;
}

const defaultLLMConnectionFilters: LLMConnectionFilters = {
  pageNumber: 1,
  pageSize: 10,
  sortBy: 'created_at',
  sortOrder: 'desc',
};

const defaultProductionConnectionFilters: ProductionConnectionFilters = {
  sortBy: 'created_at',
  sortOrder: 'desc',
  llmPlatform: '',
  llmModel: '',
};

const useStore = create<StoreState>((set) => ({
  userInfo: null,
  userId: '',
  userDomains: [],
  setUserInfo: (data) => set({ userInfo: data, userId: data?.userIdCode || '' }),
  setUserDomains: (domains: string[]) => set({ userDomains: domains }),
  llmConnectionFilters: defaultLLMConnectionFilters,
  llmConnectionPageIndex: 1,
  productionConnectionFilters: defaultProductionConnectionFilters,
  setLLMConnectionFilters: (filters) => set({ llmConnectionFilters: filters }),
  setLLMConnectionPageIndex: (pageIndex) => set({ llmConnectionPageIndex: pageIndex }),
  setProductionConnectionFilters: (filters) => set({ productionConnectionFilters: filters }),
  resetLLMConnectionFilters: () => set({ 
    llmConnectionFilters: defaultLLMConnectionFilters,
    llmConnectionPageIndex: 1,
    productionConnectionFilters: defaultProductionConnectionFilters,
  }),
}));

export default useStore;
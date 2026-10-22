import { FC, useEffect, useState } from 'react';
import { Route, Routes, useNavigate, useLocation } from 'react-router-dom';
import { Layout } from 'components';
import useStore from 'store';
import { useQuery } from '@tanstack/react-query';
import { UserInfo } from 'types/userInfo';
import { ROLES } from 'enums/roles';
import LoadingScreen from 'pages/LoadingScreen/LoadingScreen';
import LLMConnections from 'pages/LLMConnections';
import CreateLLMConnection from 'pages/LLMConnections/CreateLLMConnection';
import ViewLLMConnection from 'pages/LLMConnections/ViewLLMConnection';
import TestProductionLLM from 'pages/TestProductionLLM';
import PromptConfigurations from 'pages/PromptConfigurations';
import useTabCloseEffect from 'hooks/useTabCloseEffects';

const isLocal = import.meta.env.REACT_APP_LOCAL?.toLowerCase() === 'true';

const App: FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [hasRedirected, setHasRedirected] = useState(false);

  const { isLoading, data } = useQuery<{ response: UserInfo }>({
    queryKey: (isLocal ? ['userinfo', 'prod'] : ['auth/jwt/userinfo', 'auth']) as string[],
    onSuccess: (res) => {
      if (!isLocal) {
        localStorage.setItem('exp', res.response.JWTExpirationTimestamp);
      }
      useStore.getState().setUserInfo(res.response);
    },
  });

  const userInfo = data?.response;

  useEffect(() => {
    if (!isLoading && userInfo && !hasRedirected && location.pathname === '/') {
      const isAdmin = userInfo.authorities.some(
        (item) => item === ROLES.ROLE_ADMINISTRATOR
      );
      if (isAdmin) {
        navigate('/llm-connections');
      } else {
        navigate('/dataset-groups');
      }
      setHasRedirected(true);
    }
  }, [isLoading, userInfo, navigate, hasRedirected, location.pathname]);

  useTabCloseEffect();
  
  return (
    <>
      {isLoading ? (
        <LoadingScreen />
      ) : (
        <Routes>
          <Route element={<Layout />}>
            <Route path="/llm-connections" element={<LLMConnections />} />
            <Route path="/create-llm-connection" element={<CreateLLMConnection />} />
            <Route path="/view-llm-connection" element={<ViewLLMConnection />} />
            <Route path="/prompt-configurations" element={<PromptConfigurations />} />
            <Route path="/test-llm" element={<TestProductionLLM />} /> 


            </Route>
        </Routes>
      )}
    </>
  );
};

export default App;

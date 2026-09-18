import { FC } from 'react';
import { Outlet } from 'react-router-dom';
import useStore from 'store';
import './Layout.scss';
import { useToast } from '../../hooks/useToast';
import { MainNavigation } from '@buerokratt-ria/menu';
import { Header, useMenuCountConf } from '@buerokratt-ria/header';

const Layout: FC = () => {
    const domainBarShowing = import.meta.env.REACT_APP_ENABLE_MULTI_DOMAIN?.toLowerCase() === 'true';
    const menuCountConf = useMenuCountConf();

  return (
    <div className="layout">
      <MainNavigation countConf={menuCountConf}/>
      <div className="layout__wrapper">
        <Header 
        toastContext={useToast()} 
        user={useStore.getState().userInfo} 
        setUserDomains={useStore.getState().setUserDomains} 
        isDomainSelectorVisible={domainBarShowing}
/>
        <main className="layout__main">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;

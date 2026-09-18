import { FC, MouseEvent, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { NavLink, useLocation } from 'react-router-dom';
import { MdCorporateFare, MdFileCopy, MdKeyboardArrowDown, MdOutlineDataset, MdSearch, MdSupervisorAccount } from 'react-icons/md';
import { useQuery } from '@tanstack/react-query';
import clsx from 'clsx';
import { Icon } from 'components';
import type { MenuItem } from 'types/mainNavigation';
import './MainNavigation.scss';
import apiDev from 'services/api-dev';
import { userManagementEndpoints } from 'utils/endpoints';
import { authQueryKeys } from 'utils/queryKeys';
import { ROLES } from 'enums/roles';

const MainNavigation: FC = () => {
  const { t } = useTranslation();
  const [menuItems, setMenuItems] = useState<MenuItem[]>([]);

  const items = [
    {
      id: 'userManagement',
      label: t('menu.userManagement'),
      path: '/user-management',
      icon: <MdSupervisorAccount />,
    },
     {
      id: 'llmConnections',
      label: t('menu.llmConnections._self'),
      path: '',
      icon: <MdOutlineDataset />,
      children: [
        {
          label: t('menu.llmConnections.overview'),
          path: '/llm-connections',
        },
        {
          label: t('menu.llmConnections.promptConfigurations'),
          path: '/prompt-configurations',
        }
      ],
    },
    {
      id: 'testLLM',
      label: t('menu.testLLM'),
      path: '/test-llm',
      icon: <MdSearch />
    }
  ];

  const filterItemsByRole = (role: string[], items: MenuItem[]) => {
    return items?.filter((item) => {
      if (role.includes(ROLES.ROLE_ADMINISTRATOR)) return item?.id;
      else if (role.includes(ROLES.ROLE_MODEL_TRAINER))
        return item?.id !== 'userManagement' && item?.id !== 'integration';
      else return false;
    });
  };

  useQuery(authQueryKeys.USER_ROLES(), {
    queryFn: async () => {
      const res = await apiDev.get(userManagementEndpoints.FETCH_USER_ROLES());
      return res?.data?.response;
    },
    onSuccess: (res) => {
      const roles = res;
      const filteredItems = filterItemsByRole(roles, items);
      setMenuItems(filteredItems);
    },
    onError: (error) => {
      console.error('Error fetching user roles:', error);
    },
  });
  const location = useLocation();
  const navCollapsed = false;

  const handleNavToggle = (event: MouseEvent) => {
    const isExpanded =
      event?.currentTarget?.getAttribute('aria-expanded') === 'true';
    event?.currentTarget?.setAttribute(
      'aria-expanded',
      isExpanded ? 'false' : 'true'
    );
  };

  const renderMenuTree = (menuItems: MenuItem[]) => {
    return menuItems?.map((menuItem) => (
      <li key={menuItem?.label}>
        {menuItem?.children ? (
          <div>
            <button
              className={clsx('nav__toggle', {
                'nav__toggle--icon': !!menuItem.id,
              })}
              aria-expanded={
                menuItem?.path && location?.pathname?.includes(menuItem?.path)
                  ? 'true'
                  : 'false'
              }
              onClick={handleNavToggle}
            >
              <Icon icon={menuItem?.icon} />
              <span
                style={{
                  marginLeft: '10px',
                }}
              >
                {menuItem?.label}
              </span>
              <Icon icon={<MdKeyboardArrowDown />} />
            </button>
            <ul className="nav__submenu">
              {renderMenuTree(menuItem?.children)}
            </ul>
          </div>
        ) : (
          <NavLink to={menuItem?.path ?? '#'}>
            {' '}
            <Icon
              icon={menuItem?.icon}
              style={{
                marginRight: '10px',
              }}
            />
            {menuItem?.label}
          </NavLink>
        )}
      </li>
    ));
  };

  if (!menuItems) return null;

  return (
    <nav className={clsx('nav', { 'nav--collapsed': navCollapsed })}>
      <ul className="nav__menu">{renderMenuTree(items)}</ul>
    </nav>
  );
};

export default MainNavigation;
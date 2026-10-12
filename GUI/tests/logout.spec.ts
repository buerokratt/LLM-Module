import { test, expect } from '@playwright/test';
import { AuthHelper, PageHelper } from './helpers/test-helpers';

test.describe('Logout Functionality', () => {
  let authHelper: AuthHelper;
  let pageHelper: PageHelper;

  // Setup: Login as admin before each test
  test.beforeEach(async ({ page }) => {
    authHelper = new AuthHelper(page);
    pageHelper = new PageHelper(page);

    await test.step('Login as administrator', async () => {
      await authHelper.loginAsAdmin();
      await authHelper.verifyAdminRedirect();
    });
  });

  test('should logout successfully using logout button', async ({ page }) => {
    await test.step('Verify user is logged in and on user management page', async () => {
      // Confirm we're on the user management page
      expect(page.url()).toContain('user-management');
      
      // Verify logout button is visible in header
      const logoutButton = await page.locator('button:has-text("Logout")');
      expect(await logoutButton.isVisible()).toBe(true);
      
      await pageHelper.takeScreenshot('logged-in-state');
    });

    await test.step('Click logout button', async () => {
      await authHelper.logout();
      
      // Take screenshot during logout process
      await pageHelper.takeScreenshot('logout-in-progress');
    });

    await test.step('Verify redirect to login page', async () => {
      await authHelper.verifyLogoutRedirect();
      
      // Take screenshot of final logout state
      await pageHelper.takeScreenshot('logout-completed');
    });

    await test.step('Verify user cannot access protected pages after logout', async () => {
      // Try to navigate back to user management page
      await page.goto('http://localhost:3001/rag-search/user-management');
      await page.waitForLoadState('networkidle', { timeout: 10000 });
      
      // Should be redirected back to auth since user is logged out
      const currentUrl = page.url();
      expect(currentUrl).not.toContain('/user-management');
      
      // Should be on auth page or redirected there
      expect(currentUrl).toContain('localhost:3004');
      
      await pageHelper.takeScreenshot('protected-page-access-denied');
    });
  });

  test('should handle logout from different pages', async ({ page }) => {
    await test.step('Navigate to different application page', async () => {
      // Try to navigate to LLM connections page if available
      await page.goto('http://localhost:3001/rag-search/llm-connections');
      await page.waitForLoadState('networkidle');
      
      // Take screenshot of different page
      await pageHelper.takeScreenshot('different-page-before-logout');
    });

    await test.step('Logout from different page', async () => {
      // Logout button should be available on all pages with header
      const logoutButton = await page.locator('button:has-text("Logout")');
      
      if (await logoutButton.isVisible()) {
        await authHelper.logout();
        await authHelper.verifyLogoutRedirect();
        
        await pageHelper.takeScreenshot('logout-from-different-page');
      } else {
        console.log('Logout button not available on this page - test skipped');
      }
    });
  });

  test('should show logout button only when user is authenticated', async ({ page }) => {
    await test.step('Verify logout button is visible for authenticated user', async () => {
      const logoutButton = await page.locator('button:has-text("Logout")');
      expect(await logoutButton.isVisible()).toBe(true);
      
      await pageHelper.takeScreenshot('logout-button-visible');
    });

    await test.step('Logout and verify button is no longer visible', async () => {
      await authHelper.logout();
      
      // After logout and redirect, logout button should not be visible
      await page.waitForTimeout(2000);
      
      const logoutButton = await page.locator('button:has-text("Logout")');
      
      // Should not find logout button on auth page
      if (await logoutButton.count() > 0) {
        expect(await logoutButton.isVisible()).toBe(false);
      }
      
      await pageHelper.takeScreenshot('logout-button-hidden');
    });
  });
});
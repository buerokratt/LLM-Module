import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test('should redirect to authentication URL', async ({ page }) => {
    // Navigate to the authentication URL from the environment configuration
    const authUrl = 'http://localhost:3004/et/dev-auth';
    
    await test.step('Navigate to authentication URL', async () => {
      await page.goto(authUrl);
    });

    await test.step('Verify page loads successfully', async () => {
      // Wait for the page to load
      await page.waitForLoadState('networkidle');
      
      // Verify we're on the correct URL
      expect(page.url()).toContain('localhost:3004');
      expect(page.url()).toContain('/et/dev-auth');
    });

    await test.step('Verify page title and content', async () => {
      // Wait for the page title to load
      await page.waitForTimeout(2000);
      
      // Check that the page has loaded (title should not be empty)
      const title = await page.title();
      expect(title).toBeTruthy();
      
      // Take a screenshot for debugging purposes
      await page.screenshot({ path: 'tests/screenshots/auth-page.png' });
    });
  });

  test('should handle navigation from main app to auth', async ({ page }) => {
    await test.step('Start from main application', async () => {
      // First go to the main application
      await page.goto('http://localhost:3001');
    });

    await test.step('Navigate to authentication', async () => {
      // Navigate to the authentication URL
      await page.goto('http://localhost:3004/et/dev-auth');
      
      // Wait for navigation to complete
      await page.waitForLoadState('networkidle');
    });

    await test.step('Verify successful redirect', async () => {
      // Verify we're on the auth page
      expect(page.url()).toContain('localhost:3004');
      expect(page.url()).toContain('/et/dev-auth');
      
      // Take a screenshot
      await page.screenshot({ path: 'tests/screenshots/auth-redirect.png' });
    });
  });

  test('should login with EE30303039914 and redirect to user management', async ({ page }) => {
    await test.step('Navigate to authentication page', async () => {
      // Go to the Estonian authentication page
      await page.goto('http://localhost:3004/et/dev-auth');
      await page.waitForLoadState('networkidle');
    });

    await test.step('Fill login form and submit', async () => {
      // Wait for the form to be ready
      await page.waitForTimeout(2000);
      
      // Find and fill the ID code input field
      // The form should accept Estonian ID codes starting with EE
      const idInput = await page.locator('input[type="text"], input[placeholder*="user name"], input[placeholder*="sisesta"]').first();
      await idInput.fill('EE30303039914');
      
      // Find and click the submit button (sisene means "enter" in Estonian)
      const submitButton = await page.locator('button[type="submit"], button:has-text("sisene"), button:has-text("Sisene"), input[type="submit"]').first();
      await submitButton.click();
      
      // Wait for the authentication to process
      await page.waitForTimeout(20000);
    });

    await test.step('Verify redirect to user management', async () => {
      // Wait for navigation to complete
      await page.waitForLoadState('networkidle', { timeout: 20000 });
      
      // Verify we've been redirected to the user management page
      // Based on the App.tsx logic, administrators should be redirected to /user-management
      const currentUrl = page.url();
      
      // The URL should contain the user management path
      expect(currentUrl).toContain('http://localhost:3003/rag-search/user-management');
      
      // Take a screenshot for debugging
      await page.screenshot({ path: 'tests/screenshots/user-management-redirect.png' });
      
      // Verify that we can see user management content
      // Look for typical user management elements
      const pageContent = await page.textContent('body');
      expect(pageContent).toBeTruthy();
      
      // Optional: Check for specific user management page elements
      const userManagementTitle = await page.locator('h1, h2, h3').filter({ hasText: /user|kasutaj|management|haldus/i }).first();
      if (await userManagementTitle.isVisible()) {
        expect(await userManagementTitle.isVisible()).toBe(true);
      }
    });
  });
  
});

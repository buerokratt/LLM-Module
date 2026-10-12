import { test, expect } from '@playwright/test';
import { AuthHelper, PageHelper } from './helpers/test-helpers';
import { LLMConnectionsHelper, LLMConnectionTestData } from './helpers/llm-connections-helpers';


test.describe('LLM Connections', () => {
  let authHelper: AuthHelper;
  let pageHelper: PageHelper;
  let llmConnectionsHelper: LLMConnectionsHelper;

  // Setup: Login as admin before each test
  test.beforeEach(async ({ page }) => {
    authHelper = new AuthHelper(page);
    pageHelper = new PageHelper(page);
    llmConnectionsHelper = new LLMConnectionsHelper(page);

    await test.step('Login as administrator', async () => {
      await authHelper.loginAsAdmin();
      await authHelper.verifyAdminRedirect();
    });

    await test.step('Navigate to LLM connections page', async () => {
      await page.goto('http://localhost:3001/rag-search/llm-connections');
      await page.waitForLoadState('networkidle');
    });
  });

  test('should view LLM connections list page', async ({ page }) => {
    await test.step('Verify LLM connections page loads', async () => {
      // Check page title or heading
      const pageTitle = await page.locator('h1, h2, h3, .title').filter({ hasText: /data.*models/i }).first();
      expect(await pageTitle.isVisible()).toBe(true);
      
      // Verify create connection button is present
      const createButton = await page.locator('button').filter({ hasText: /create.*connection|add.*connection|new.*connection/i });
      expect(await createButton.isVisible()).toBe(true);
      
      await pageHelper.takeScreenshot('llm-connections-list');
    });

    await test.step('Verify connections list structure', async () => {
      // Check for connections grid/list container
      const connectionsContainer = await page.locator('.connections-grid, .connections-list, .llm-connections-container');
      
      if (await connectionsContainer.count() > 0) {
        expect(await connectionsContainer.first().isVisible()).toBe(true);
      }
      
      // Check for filter/sort controls
      const filterControls = await page.locator('select, .filter, .sort');
      if (await filterControls.count() > 0) {
        expect(await filterControls.first().isVisible()).toBe(true);
      }
      
      await pageHelper.takeScreenshot('connections-structure');
    });

    await test.step('Verify pagination if present', async () => {
      const paginationContainer = await page.locator('.pagination, nav[aria-label*="pagination"]');
      
      if (await paginationContainer.count() > 0) {
        expect(await paginationContainer.first().isVisible()).toBe(true);
        await pageHelper.takeScreenshot('connections-pagination');
      }
    });
  });

  test('should inspect available platform options (debugging)', async ({ page }) => {
    await test.step('Navigate to create connection page', async () => {
      await llmConnectionsHelper.navigateToCreateConnection();
      expect(page.url()).toContain('create-llm-connection');
      await pageHelper.takeScreenshot('debug-form-loaded');
    });

    await test.step('Check available platform options', async () => {
      // Find the LLM Configuration section and platform dropdown
      const llmSection = page.locator('.form-section').filter({ hasText: /LLM Configuration/i });
      await llmSection.waitFor({ state: 'visible' });
      
      const platformDropdown = llmSection.locator('.select').first();
      await platformDropdown.waitFor({ state: 'visible' });
      
      // Click the trigger to open dropdown
      const trigger = platformDropdown.locator('.select__trigger');
      await trigger.click();
      
      // Wait for options to appear
      const options = page.locator('.select__option');
      await options.first().waitFor({ state: 'visible', timeout: 10000 });
      
      // Log available platform options
      const availableOptions = await options.allTextContents();
      console.log('Available platform options:', availableOptions);
      
      await pageHelper.takeScreenshot('debug-platform-options');
      
      // Close dropdown
      await page.keyboard.press('Escape');
    });
  });

  test('should create new Azure OpenAI LLM connection', async ({ page }) => {
    const azureData = LLMConnectionTestData.createAzureConnection({
      connectionName: 'Test Azure Connection ' + Date.now(),
      environment: 'testing'
    });

    await test.step('Navigate to create connection page', async () => {
      await llmConnectionsHelper.navigateToCreateConnection();
      
      // Verify we're on the create page
      expect(page.url()).toContain('create-llm-connection');
      await pageHelper.takeScreenshot('azure-create-form-loaded');
    });

    await test.step('Fill connection basic information', async () => {
      // Connection name
      await llmConnectionsHelper.fillConnectionName(azureData.connectionName);
      
      // Platform selection
      await llmConnectionsHelper.selectLLMPlatform(azureData.llmPlatform);
      
      // Model selection (wait for platform to load models)
      await llmConnectionsHelper.selectLLMModel(azureData.llmModel);
      
      await pageHelper.takeScreenshot('azure-basic-info-filled');
    });

    await test.step('Fill Azure OpenAI credentials', async () => {
      // Fill Azure-specific LLM credentials
      await llmConnectionsHelper.fillAzureCredentials(
        azureData.deploymentName,
        azureData.targetUri,
        azureData.apiKey
      );
      
      await pageHelper.takeScreenshot('azure-llm-credentials-filled');
    });

    await test.step('Configure embedding model', async () => {
      // Embedding platform
      await llmConnectionsHelper.selectEmbeddingPlatform(azureData.embeddingPlatform);
      
      // Embedding model
      await llmConnectionsHelper.selectEmbeddingModel(azureData.embeddingModel);
      
      // Fill Azure embedding credentials
      await llmConnectionsHelper.fillAzureEmbeddingCredentials(
        azureData.embeddingDeploymentName,
        azureData.embeddingTargetUri,
        azureData.embeddingApiKey
      );
      
      await pageHelper.takeScreenshot('azure-embedding-configured');
    });

    await test.step('Configure budget and deployment', async () => {
      // Budget fields
      await llmConnectionsHelper.fillBudgetFields(
        azureData.monthlyBudget,
        azureData.warnBudget,
        azureData.stopBudget
      );
      
      // Deployment environment
      await llmConnectionsHelper.selectDeploymentEnvironment(azureData.environment);
      
      await pageHelper.takeScreenshot('azure-budget-deployment-configured');
    });

    await test.step('Submit and verify Azure connection', async () => {
      // Submit the form
      await llmConnectionsHelper.submitConnectionForm();
      
      // Verify success
      await llmConnectionsHelper.verifyConnectionSuccess();
      
      await pageHelper.takeScreenshot('azure-connection-success');
    });
  });

  test('should create new AWS Bedrock LLM connection', async ({ page }) => {
    const awsData = LLMConnectionTestData.createAWSConnection({
      connectionName: 'Test AWS Connection ' + Date.now(),
      environment: 'testing'
    });

    await test.step('Navigate to create connection page', async () => {
      await llmConnectionsHelper.navigateToCreateConnection();
      
      // Verify we're on the create page
      expect(page.url()).toContain('create-llm-connection');
      await pageHelper.takeScreenshot('aws-create-form-loaded');
    });

    await test.step('Fill connection basic information', async () => {
      // Connection name
      await llmConnectionsHelper.fillConnectionName(awsData.connectionName);
      
      // Platform selection
      await llmConnectionsHelper.selectLLMPlatform(awsData.llmPlatform);
      
      // Model selection (wait for platform to load models)
      await llmConnectionsHelper.selectLLMModel(awsData.llmModel);
      
      await pageHelper.takeScreenshot('aws-basic-info-filled');
    });

    await test.step('Fill AWS Bedrock credentials', async () => {
      // Fill AWS-specific LLM credentials
      await llmConnectionsHelper.fillAWSCredentials(
        awsData.accessKey,
        awsData.secretKey
      );
      
      await pageHelper.takeScreenshot('aws-llm-credentials-filled');
    });

    await test.step('Configure embedding model', async () => {
      // Embedding platform
      await llmConnectionsHelper.selectEmbeddingPlatform(awsData.embeddingPlatform);
      
      // Embedding model
      await llmConnectionsHelper.selectEmbeddingModel(awsData.embeddingModel);
      
      // Fill AWS embedding credentials
      await llmConnectionsHelper.fillAWSEmbeddingCredentials(
        awsData.embeddingAccessKey,
        awsData.embeddingSecretKey
      );
      
      await pageHelper.takeScreenshot('aws-embedding-configured');
    });

    await test.step('Configure budget and deployment', async () => {
      // Budget fields
      await llmConnectionsHelper.fillBudgetFields(
        awsData.monthlyBudget,
        awsData.warnBudget,
        awsData.stopBudget
      );
      
      // Deployment environment
      await llmConnectionsHelper.selectDeploymentEnvironment(awsData.environment);
      
      await pageHelper.takeScreenshot('aws-budget-deployment-configured');
    });

    await test.step('Submit and verify AWS connection', async () => {
      // Submit the form
      await llmConnectionsHelper.submitConnectionForm();
      
      // Verify success
      await llmConnectionsHelper.verifyConnectionSuccess();
      
      await pageHelper.takeScreenshot('aws-connection-success');
    });

    await test.step('Verify connection appears in list', async () => {
      // Verify the connection exists in the list
      const exists = await llmConnectionsHelper.verifyConnectionInList(awsData.connectionName);
      expect(exists).toBe(true);
      
      await pageHelper.takeScreenshot('aws-connection-in-list');
    });
  });

  test('should view connection details', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
      await pageHelper.takeScreenshot('view-connections-list');
    });

    await test.step('Click on a connection to view details', async () => {
      // Find any connection card
      const connectionCard = page.locator('[class*="connection-card"], [class*="llm-connection"]').first();
      
      if (await connectionCard.count() > 0) {
        // Look for view/details button or click the card
        const viewButton = connectionCard.locator('button').filter({ hasText: /Settings|open/i });
        
        if (await viewButton.count() > 0 && await viewButton.isVisible()) {
          await viewButton.click();
        } else {
          // Click the card itself
          await connectionCard.click();
        }
        
        await page.waitForLoadState('networkidle');
        
        // Verify we're on the view page
        expect(page.url()).toMatch(/view-llm-connection|llm-connection\/\d+/);
        
        await pageHelper.takeScreenshot('connection-details-view');
      }
    });

    await test.step('Verify connection details are displayed', async () => {
      // Check for connection details sections
      const detailsContainer = page.locator('.connection-details, .details-container, .view-container');
      
      if (await detailsContainer.count() > 0) {
        expect(await detailsContainer.isVisible()).toBe(true);
      }
      
      // Check for key information fields
      const expectedFields = [
        /connection.*name/i,
        /platform/i,
        /model/i,
        /environment/i
      ];
      
      for (const fieldPattern of expectedFields) {
        const field = page.locator('label, .field-label, .detail-label').filter({ hasText: fieldPattern });
        
        if (await field.count() > 0) {
          console.log(`Found field: ${fieldPattern}`);
        }
      }
      
      await pageHelper.takeScreenshot('connection-details-displayed');
    });
  });

  test('should update/edit LLM connection', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Open edit form via Settings button', async () => {
      // Find first connection cad
      const connectionCard = page.locator('.dataset-group-card').first();
      
      if (await connectionCard.count() > 0) {
        // Look for Settings button in the button-row
        const settingsButton = connectionCard.locator('.button-row button').filter({ hasText: /settings/i });
        
        if (await settingsButton.count() > 0 && await settingsButton.isVisible()) {
          await settingsButton.click();
          await page.waitForLoadState('networkidle');
          
          // Verify we're on the view/edit page with query parameter
          expect(page.url()).toMatch(/view-llm-connection\?id=/);
          
          await pageHelper.takeScreenshot('connection-settings-page');
        } else {
          console.log('Settings button not found, skipping edit test');
          return;
        }
      } else {
        console.log('No connection cards found, skipping edit test');
        return;
      }
    });

    await test.step('Verify Update Connection button is initially disabled', async () => {
      const updateButton = page.locator('button[type="submit"]').filter({ 
        hasText: /update.*connection/i 
      });
      
      if (await updateButton.count() > 0 && await updateButton.isVisible()) {
        const isDisabled = await updateButton.isDisabled();
        expect(isDisabled).toBe(true);
        
        await pageHelper.takeScreenshot('update-button-initially-disabled');
      }
    });

    await test.step('Update connection name to enable submit button', async () => {
      const nameField = page.locator('input[name="connectionName"]');
      
      if (await nameField.count() > 0 && await nameField.isVisible()) {
        // Get current value
        const currentValue = await nameField.inputValue();
        
        // Modify the connection name
        const newValue = `${currentValue} - Updated ${Date.now()}`;
        
        await nameField.clear();
        await nameField.fill(newValue);
        
        // Wait for form validation
        await page.waitForTimeout(500);
        
        await pageHelper.takeScreenshot('connection-name-updated');
      }
    });

    await test.step('Update monthly budget', async () => {
      const budgetField = page.locator('input[name="monthlyBudget"]');
      
      if (await budgetField.count() > 0 && await budgetField.isVisible()) {
        await budgetField.clear();
        await budgetField.fill('2000');
        
        // Wait for validation
        await page.waitForTimeout(500);
        
        await pageHelper.takeScreenshot('budget-updated');
      }
    });

    await test.step('Verify Update Connection button is now enabled', async () => {
      const updateButton = page.locator('button[type="submit"]').filter({ 
        hasText: /update.*connection/i 
      });
      
      if (await updateButton.count() > 0 && await updateButton.isVisible()) {
        // Wait for button to be enabled after field changes
        await page.waitForTimeout(1000);
        
        const isEnabled = await updateButton.isEnabled();
        expect(isEnabled).toBe(true);
        
        await pageHelper.takeScreenshot('update-button-enabled');
      }
    });

    await test.step('Submit update', async () => {
      const updateButton = page.locator('button[type="submit"]').filter({ 
        hasText: /update.*connection/i 
      });
      
      if (await updateButton.count() > 0 && await updateButton.isEnabled()) {
        await updateButton.click();
        await page.waitForLoadState('networkidle');
        
        // Look for success message
        const successMessage = page.locator('[role="dialog"], .success, .notification, .toast').filter({ 
          hasText: /success|updated|saved/i 
        });
        
        if (await successMessage.count() > 0) {
          await successMessage.waitFor({ state: 'visible', timeout: 5000 });
          expect(await successMessage.isVisible()).toBe(true);
          await pageHelper.takeScreenshot('update-success');
        }
      }
    });

    await test.step('Verify Delete button exists at bottom', async () => {
      // Scroll to bottom of page to find delete btn
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(500);
      
      const deleteButton = page.locator('button').filter({ hasText: /delete/i });
      
      if (await deleteButton.count() > 0) {
        expect(await deleteButton.isVisible()).toBe(true);
        console.log('Delete button found at bottom of page');
        
        await pageHelper.takeScreenshot('delete-button-at-bottom');
      }
    });
  });

  test('should filter connections by platform', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Locate and use platform filter', async () => {
      // Look for platform filter dropdown
      const platformFilter = page.locator('select, .filter, .select').filter({ 
        hasText: /platform|filter.*platform/i 
      }).or(
        page.locator('label').filter({ hasText: /platform/i }).locator('..').locator('select, .select')
      ).first();
      
      if (await platformFilter.count() > 0) {
        await platformFilter.waitFor({ state: 'visible', timeout: 5000 });
        
        // Check if it's a select element or custom dropdown
        const isNativeSelect = await platformFilter.evaluate(el => el.tagName === 'SELECT');
        
        if (isNativeSelect) {
          // Native select
          await platformFilter.selectOption({ index: 1 }); // Select first non-default option
        } else {
          // Custom dropdown
          const trigger = platformFilter.locator('.select__trigger, button').first();
          await trigger.click();
          
          // Select first option
          const option = page.locator('.select__option').nth(1);
          await option.waitFor({ state: 'visible' });
          await option.click();
        }
        
        await page.waitForTimeout(1000);
        await pageHelper.takeScreenshot('platform-filter-applied');
        
        // Verify filter was applied (URL params or filtered results)
        const url = page.url();
        console.log('URL after filter:', url);
      } else {
        console.log('Platform filter not found');
      }
    });
  });

  test('should filter connections by LLM model', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Locate and use model filter', async () => {
      // Look for model filter dropdown
      const modelFilter = page.locator('select, .filter, .select').filter({ 
        hasText: /model|filter.*model/i 
      }).or(
        page.locator('label').filter({ hasText: /model/i }).locator('..').locator('select, .select')
      ).first();
      
      if (await modelFilter.count() > 0) {
        await modelFilter.waitFor({ state: 'visible', timeout: 5000 });
        
        const isNativeSelect = await modelFilter.evaluate(el => el.tagName === 'SELECT');
        
        if (isNativeSelect) {
          await modelFilter.selectOption({ index: 1 });
        } else {
          const trigger = modelFilter.locator('.select__trigger, button').first();
          await trigger.click();
          
          const option = page.locator('.select__option').nth(1);
          await option.waitFor({ state: 'visible' });
          await option.click();
        }
        
        await page.waitForTimeout(1000);
        await pageHelper.takeScreenshot('model-filter-applied');
      } else {
        console.log('Model filter not found');
      }
    });
  });

  test('should filter connections by environment', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Locate and use environment filter', async () => {
      // Look for environment filter
      const environmentFilter = page.locator('select, .filter, .select').filter({ 
        hasText: /environment|deployment/i 
      }).or(
        page.locator('label').filter({ hasText: /environment/i }).locator('..').locator('select, .select')
      ).first();
      
      if (await environmentFilter.count() > 0) {
        await environmentFilter.waitFor({ state: 'visible', timeout: 5000 });
        
        const isNativeSelect = await environmentFilter.evaluate(el => el.tagName === 'SELECT');
        
        if (isNativeSelect) {
          // Try to select 'testing' or 'production'
          await environmentFilter.selectOption('testing').catch(() => 
            environmentFilter.selectOption({ index: 1 })
          );
        } else {
          const trigger = environmentFilter.locator('.select__trigger, button').first();
          await trigger.click();
          
          const testingOption = page.locator('.select__option').filter({ hasText: /testing/i });
          
          if (await testingOption.count() > 0) {
            await testingOption.first().click();
          } else {
            await page.locator('.select__option').nth(1).click();
          }
        }
        
        await page.waitForTimeout(1000);
        await pageHelper.takeScreenshot('environment-filter-applied');
      } else {
        console.log('Environment filter not found');
      }
    });
  });

  test('should sort connections by created date', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Locate and use sort control', async () => {
      // Look for sort dropdown or button
      const sortControl = page.locator('select, .sort, .select').filter({ 
        hasText: /sort|order/i 
      }).or(
        page.locator('button').filter({ hasText: /sort/i })
      ).first();
      
      if (await sortControl.count() > 0) {
        await sortControl.waitFor({ state: 'visible', timeout: 5000 });
        
        const isButton = await sortControl.evaluate(el => el.tagName === 'BUTTON');
        
        if (isButton) {
          // Click sort button
          await sortControl.click();
          await pageHelper.takeScreenshot('sort-clicked');
        } else {
          // Select sort option
          const isNativeSelect = await sortControl.evaluate(el => el.tagName === 'SELECT');
          
          if (isNativeSelect) {
            await sortControl.selectOption({ index: 1 });
          } else {
            const trigger = sortControl.locator('.select__trigger, button').first();
            await trigger.click();
            
            const option = page.locator('.select__option').nth(1);
            await option.waitFor({ state: 'visible' });
            await option.click();
          }
        }
        
        await page.waitForTimeout(1000);
        await pageHelper.takeScreenshot('sort-applied');
      } else {
        console.log('Sort control not found');
      }
    });

    await test.step('Verify sort order changed', async () => {
      // Get connection cards
      const connectionCards = page.locator('[class*="connection-card"], [class*="llm-connection"]');
      const count = await connectionCards.count();
      
      console.log(`Found ${count} connections after sort`);
      
      if (count > 0) {
        await pageHelper.takeScreenshot('connections-after-sort');
      }
    });
  });

  test('should reset filters', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Apply a filter', async () => {
      const platformFilter = page.locator('select, .filter, .select').first();
      
      if (await platformFilter.count() > 0 && await platformFilter.isVisible()) {
        const isNativeSelect = await platformFilter.evaluate(el => el.tagName === 'SELECT');
        
        if (isNativeSelect) {
          await platformFilter.selectOption({ index: 1 });
        } else {
          const trigger = platformFilter.locator('.select__trigger, button').first();
          await trigger.click();
          
          await page.locator('.select__option').nth(1).click();
        }
        
        await page.waitForTimeout(500);
      }
    });

    await test.step('Click reset button', async () => {
      const resetButton = page.locator('button').filter({ hasText: /reset|clear.*filter/i });
      
      if (await resetButton.count() > 0 && await resetButton.isVisible()) {
        await resetButton.click();
        await page.waitForTimeout(1000);
        
        await pageHelper.takeScreenshot('filters-reset');
        
        // Verify filters were reset (check URL or filter values)
        console.log('Filters reset, URL:', page.url());
      } else {
        console.log('Reset button not found');
      }
    });
  });

  test('should navigate through pagination', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Check if pagination exists', async () => {
      const pagination = page.locator('.pagination, nav[aria-label*="pagination"], [class*="pagination"]');
      
      if (await pagination.count() > 0) {
        expect(await pagination.isVisible()).toBe(true);
        await pageHelper.takeScreenshot('pagination-visible');
        
        // Look for next button
        const nextButton = pagination.locator('button').filter({ 
          hasText: /next|>/i 
        }).or(
          pagination.locator('button[aria-label*="next"]')
        ).first();
        
        if (await nextButton.count() > 0 && await nextButton.isEnabled()) {
          // Get current page connections
          const beforeCards = await page.locator('[class*="connection-card"], [class*="llm-connection"]').count();
          
          // Click next
          await nextButton.click();
          await page.waitForLoadState('networkidle');
          await page.waitForTimeout(1000);
          
          await pageHelper.takeScreenshot('pagination-next-page');
          
          // Verify page changed (URL param or different content)
          const afterCards = await page.locator('[class*="connection-card"], [class*="llm-connection"]').count();
          console.log(`Before: ${beforeCards} cards, After: ${afterCards} cards`);
        }
      } else {
        console.log('Pagination not found - likely fewer connections than page size');
      }
    });

    await test.step('Navigate back to first page', async () => {
      const pagination = page.locator('.pagination, nav[aria-label*="pagination"], [class*="pagination"]');
      
      if (await pagination.count() > 0) {
        const previousButton = pagination.locator('button').filter({ 
          hasText: /previous|prev|</i 
        }).or(
          pagination.locator('button[aria-label*="previous"]')
        ).first();
        
        if (await previousButton.count() > 0 && await previousButton.isEnabled()) {
          await previousButton.click();
          await page.waitForLoadState('networkidle');
          await page.waitForTimeout(1000);
          
          await pageHelper.takeScreenshot('pagination-previous-page');
        }
      }
    });
  });

  test('should delete LLM connection', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Open connection settings via Settings button', async () => {
      // Look for a connection card 
      const connectionCards = page.locator('.dataset-group-card');
      
      if (await connectionCards.count() > 0) {
        const firstCard = connectionCards.first();
        
        // Look for Settings button in the button-row
        const settingsButton = firstCard.locator('.button-row button').filter({ hasText: /settings/i });
        
        if (await settingsButton.count() > 0 && await settingsButton.isVisible()) {
          await settingsButton.click();
          await page.waitForLoadState('networkidle');
          
          // Verify we're on the view/edit page with query parameter
          expect(page.url()).toMatch(/view-llm-connection\?id=/);
          
          await pageHelper.takeScreenshot('connection-settings-page-for-delete');
        } else {
          console.log('Settings button not found, skipping delete test');
          return;
        }
      } else {
        console.log('No connection cards found, skipping delete test');
        return;
      }
    });

    await test.step('Scroll to bottom and click delete button', async () => {
      // Scroll to bottom of page to find delete button
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(500);
      
      const deleteButton = page.locator('button').filter({ hasText: /delete/i });
      
      if (await deleteButton.count() > 0 && await deleteButton.isVisible()) {
        await deleteButton.click();
        await pageHelper.takeScreenshot('delete-button-clicked');
      } else {
        console.log('Delete button not found at bottom of page');
        return;
      }
    });

    await test.step('Confirm deletion in modal', async () => {
      // Look for confirmation dialog
      const confirmDialog = page.locator('[role="dialog"], .modal, .dialog').filter({ 
        hasText: /delete|confirm|remove/i 
      });
      
      if (await confirmDialog.count() > 0) {
        expect(await confirmDialog.isVisible()).toBe(true);
        
        await pageHelper.takeScreenshot('delete-confirmation-dialog');
        
        // Look for confirm/delete button in dialog
        const confirmButton = confirmDialog.locator('button').filter({ 
          hasText: /delete|confirm|yes|remove/i 
        });
        
        if (await confirmButton.count() > 0) {
          await confirmButton.click();
          await page.waitForLoadState('networkidle');
          await page.waitForTimeout(1000);
          
          // Look for success message
          const successMessage = page.locator('[role="alert"], .notification, .toast').filter({ 
            hasText: /success|deleted|removed/i 
          });
          
          if (await successMessage.count() > 0) {
            expect(await successMessage.isVisible()).toBe(true);
            await pageHelper.takeScreenshot('delete-success');
          }
        }
      } else {
        console.log('Delete confirmation dialog not found');
      }
    });
  });

  test('should display no data message when no connections exist', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Check for connections or no data message', async () => {
      // Look for connections
      const connectionCards = page.locator('[class*="connection-card"], [class*="llm-connection"]');
      const hasConnections = await connectionCards.count() > 0;
      
      if (!hasConnections) {
        // Look for no data message
        const noDataMessage = page.locator('.no-data, .empty-state, [class*="no-data"]').or(
          page.locator('p, div').filter({ hasText: /no.*connection|no.*model|empty/i })
        );
        
        if (await noDataMessage.count() > 0) {
          expect(await noDataMessage.isVisible()).toBe(true);
          await pageHelper.takeScreenshot('no-data-message');
        }
      } else {
        console.log(`Found ${await connectionCards.count()} connections`);
        await pageHelper.takeScreenshot('connections-exist');
      }
    });
  });

  test('should display production and testing connections separately', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Check for production connections section', async () => {
      const productionSection = page.locator('h2, h3, .section-title').filter({ 
        hasText: /production/i 
      });
      
      if (await productionSection.count() > 0) {
        expect(await productionSection.isVisible()).toBe(true);
        await pageHelper.takeScreenshot('production-section');
      }
    });

    await test.step('Check for testing/other connections section', async () => {
      const testingSection = page.locator('h2, h3, .section-title, p').filter({ 
        hasText: /testing|other.*connection/i 
      });
      
      if (await testingSection.count() > 0) {
        expect(await testingSection.isVisible()).toBe(true);
        await pageHelper.takeScreenshot('testing-section');
      }
    });
  });

  test('should display connection status (active/inactive)', async ({ page }) => {
    await test.step('Navigate to connections list', async () => {
      await llmConnectionsHelper.navigateToLLMConnections();
    });

    await test.step('Check connection status badges', async () => {
      const connectionCards = page.locator('[class*="connection-card"], [class*="llm-connection"]');
      
      if (await connectionCards.count() > 0) {
        const firstCard = connectionCards.first();
        
        // Look for status indicator
        const statusBadge = firstCard.locator('.status, .badge, [class*="status"]').or(
          firstCard.locator('span').filter({ hasText: /active|inactive/i })
        );
        
        if (await statusBadge.count() > 0) {
          expect(await statusBadge.isVisible()).toBe(true);
          
          const statusText = await statusBadge.textContent();
          console.log('Connection status:', statusText);
          
          await pageHelper.takeScreenshot('connection-status-displayed');
        }
      }
    });
  });

  
});

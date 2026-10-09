import { Page, defineConfig } from '@playwright/test';

/**
 * LLM Connections Helper
 */

export default defineConfig({
  use: {
    // This tells Playwright to automatically capture a screenshot whenever a test fails
    screenshot: 'only-on-failure',
  },
});

export class LLMConnectionsHelper {
  constructor(private page: Page) {}

  async navigateToLLMConnections(): Promise<void> {
    await this.page.goto('http://localhost:3003/rag-search/llm-connections');
    
    const header = this.page.locator('div.title').filter({ hasText: /LLM connections/i });
    await header.waitFor({ state: 'visible', timeout: 10000 });
  }

  async navigateToCreateConnection(): Promise<void> {
    await this.navigateToLLMConnections();

    const createButton = this.page.locator('button').filter({
      hasText: /create.*connection|add.*connection|new.*connection/i
    });

    if (await createButton.count() > 0 && await createButton.isVisible()) {
      await createButton.click();
    } else {
      await this.page.goto('http://localhost:3003/rag-search/create-llm-connection');
    }

    await this.page.locator('input[name="connectionName"]').waitFor({ state: 'visible', timeout: 15000 });

    // Wait for platform dropdown to finish loading from API (disappear disabled/loading state)
    await this.page.locator('.form-section').filter({ hasText: /LLM Configuration/i })
      .locator('.select').first()
      .locator('.select__trigger:not([disabled])')
      .waitFor({ state: 'attached', timeout: 15000 });
  }

  /**
   * Fill connection name field
   */
  async fillConnectionName(name: string): Promise<void> {
    const nameField = this.page.locator('input[name="connectionName"]');
    await nameField.waitFor({ state: 'visible' });
    await nameField.fill(name);
  }

  /**
   * Select LLM platform from dropdown
   */
 async selectLLMPlatform(platformLabel: string): Promise<void> {
  const llmSection = this.page.locator('.form-section').filter({ hasText: /LLM Configuration/i });
  await llmSection.waitFor({ state: 'visible', timeout: 5000 });

  const platformDropdown = llmSection.locator('.select').first();
  // Wait for loading state to clear (API data ready)
  await platformDropdown.locator('.select__trigger:not([disabled])').waitFor({ state: 'attached', timeout: 10000 });

  const trigger = platformDropdown.locator('.select__trigger').first();
  await trigger.click();

  // Options render inside the listbox as li elements
  const options = platformDropdown.locator('[role="listbox"] li, [role="option"]');
  await options.first().waitFor({ state: 'visible', timeout: 10000 });

  const targetOption = options.filter({ hasText: new RegExp(platformLabel, 'i') });
  if (await targetOption.count() === 0) {
    const availableOptions = await options.allTextContents();
    throw new Error(`Platform "${platformLabel}" not found. Available options: ${availableOptions.join(', ')}`);
  }
  await targetOption.first().click();
  }
  

  /**
   * Select LLM model from dropdown
   */
async selectLLMModel(modelLabel: string): Promise<void> {
  const llmSection = this.page.locator('.form-section').filter({ hasText: /LLM Configuration/i });
  const modelDropdown = llmSection.locator('.select').nth(1);

  // Wait for model dropdown to finish loading after platform was selected
  await modelDropdown.locator('.select__trigger:not([disabled])').waitFor({ state: 'attached', timeout: 15000 });

  const trigger = modelDropdown.locator('.select__trigger').first();
  await trigger.click();

  const options = modelDropdown.locator('[role="listbox"] li, [role="option"]');
  await options.first().waitFor({ state: 'visible', timeout: 8000 });

  const targetOption = options.filter({ hasText: new RegExp(modelLabel, 'i') });
  if (await targetOption.count() === 0) {
    const availableOptions = await options.allTextContents();
    throw new Error(`Model "${modelLabel}" not found. Available options: ${availableOptions.join(', ')}`);
  }
  await targetOption.first().click();
}

  /**
   * Select embedding platform from dropdown
   */
  async selectEmbeddingPlatform(platformLabel: string): Promise<void> {
    const embeddingSection = this.page.locator('.form-section').filter({ hasText: /Embedding Model Configuration/i });
    await embeddingSection.waitFor({ state: 'visible', timeout: 5000 });

    const platformDropdown = embeddingSection.locator('.select').first();
    await platformDropdown.locator('.select__trigger:not([disabled])').waitFor({ state: 'attached', timeout: 10000 });

    const trigger = platformDropdown.locator('.select__trigger').first();
    await trigger.click();

    const options = platformDropdown.locator('[role="listbox"] li, [role="option"]');
    await options.first().waitFor({ state: 'visible', timeout: 8000 });

    const targetOption = options.filter({ hasText: new RegExp(platformLabel, 'i') });
    if (await targetOption.count() === 0) {
      const availableOptions = await options.allTextContents();
      throw new Error(`Embedding platform "${platformLabel}" not found. Available options: ${availableOptions.join(', ')}`);
    }
    await targetOption.first().click();

    // Wait for embedding model dropdown to become enabled after platform change
    const embeddingModelDropdown = embeddingSection.locator('.select').nth(1);
    await embeddingModelDropdown.locator('.select__trigger:not([disabled])').waitFor({ state: 'attached', timeout: 20000 });
  }

  /**
   * Select embedding model from dropdown
   */
  async selectEmbeddingModel(modelLabel: string): Promise<void> {
    const embeddingSection = this.page.locator('.form-section').filter({ hasText: /Embedding Model Configuration/i });
    await embeddingSection.waitFor({ state: 'visible', timeout: 5000 });

    const modelDropdown = embeddingSection.locator('.select').nth(1);
    await modelDropdown.locator('.select__trigger:not([disabled])').waitFor({ state: 'attached', timeout: 10000 });

    const trigger = modelDropdown.locator('.select__trigger').first();
    await trigger.click();

    const options = modelDropdown.locator('[role="listbox"] li, [role="option"]');
    await options.first().waitFor({ state: 'visible', timeout: 8000 });

    const targetOption = options.filter({ hasText: new RegExp(modelLabel, 'i') });
    if (await targetOption.count() === 0) {
      const availableOptions = await options.allTextContents();
      throw new Error(`Embedding model "${modelLabel}" not found. Available options: ${availableOptions.join(', ')}`);
    }
    await targetOption.first().click();
  }

  /**
   * Fill budget and threshold fields
   */
  async fillBudgetFields(monthlyBudget: string, warnBudget: string, stopBudget?: string): Promise<void> {
    // Monthly budget
    const monthlyBudgetField = this.page.locator('input[name="monthlyBudget"]');
    await monthlyBudgetField.fill(monthlyBudget);
    
    // Warn budget (percentage - remove % if provided)
    const warnBudgetField = this.page.locator('input[name="warnBudget"]');
    const warnValue = warnBudget.replace('%', '');
    await warnBudgetField.fill(warnValue);
    
    // If stop budget is provided and disconnect checkbox needs to be checked
    if (stopBudget) {
      // Try multiple approaches to check the checkbox
      const disconnectCheckbox = this.page.locator('input[name="disconnectOnBudgetExceed"]');
      
      // First, try to find and click the label associated with the checkbox
      const checkboxLabel = this.page.locator('label').filter({ 
        has: disconnectCheckbox 
      }).or(
        this.page.locator('label[for]:has-text("disconnect")').or(
          this.page.locator('label:has-text("Disconnect")').or(
            this.page.locator('label:has-text("budget exceed")')
          )
        )
      );
      
      if (await checkboxLabel.count() > 0 && await checkboxLabel.first().isVisible()) {
        // Click the label instead of the checkbox
        await checkboxLabel.first().click();
      } else {
        // Fallback: force check the checkbox even if not visible
        await disconnectCheckbox.check({ force: true });
      }
      
      // Wait for stop budget field to appear
      await this.page.waitForTimeout(1000);
      
      const stopBudgetField = this.page.locator('input[name="stopBudget"]');
      await stopBudgetField.waitFor({ state: 'visible', timeout: 5000 });
      const stopValue = stopBudget.replace('%', '');
      await stopBudgetField.fill(stopValue);
    }
  }

  /**
   * Select deployment environment using radio buttons
   */
  async selectDeploymentEnvironment(environment: 'testing' | 'production'): Promise<void> {
    const radioOption = this.page.locator(`input[type="radio"][value="${environment}"]`);
    await radioOption.check();
  }

  /**
   * Fill Azure OpenAI specific credentials
   */
  async fillAzureCredentials(deploymentName: string, targetUri: string, apiKey: string): Promise<void> {
    // Deployment name
    const deploymentField = this.page.locator('input[name="deploymentName"]');
    await deploymentField.waitFor({ state: 'visible' });
    await deploymentField.fill(deploymentName);
    
    // Target URI
    const uriField = this.page.locator('input[name="targetUri"]');
    await uriField.fill(targetUri);
    
    // API Key
    const apiKeyField = this.page.locator('input[name="apiKey"]');
    await apiKeyField.fill(apiKey);
  }

  /**
   * Fill Azure OpenAI embedding credentials
   */
  async fillAzureEmbeddingCredentials(deploymentName: string, targetUri: string, apiKey: string): Promise<void> {
    // Embedding deployment name
    const embeddingDeploymentField = this.page.locator('input[name="embeddingDeploymentName"]');
    await embeddingDeploymentField.waitFor({ state: 'visible' });
    await embeddingDeploymentField.fill(deploymentName);
    
    // Embedding target URI
    const embeddingUriField = this.page.locator('input[name="embeddingTargetUri"]');
    await embeddingUriField.fill(targetUri);
    
    // Embedding API Key
    const embeddingApiKeyField = this.page.locator('input[name="embeddingAzureApiKey"]');
    await embeddingApiKeyField.fill(apiKey);
  }

  /**
   * Fill AWS Bedrock specific credentials
   */
  async fillAWSCredentials(accessKey: string, secretKey: string): Promise<void> {
    // Access key
    const accessKeyField = this.page.locator('input[name="accessKey"]');
    await accessKeyField.waitFor({ state: 'visible' });
    await accessKeyField.fill(accessKey);
    
    // Secret key
    const secretKeyField = this.page.locator('input[name="secretKey"]');
    await secretKeyField.fill(secretKey);
  }

  /**
   * Fill AWS Bedrock embedding credentials
   */
  async fillAWSEmbeddingCredentials(accessKey: string, secretKey: string): Promise<void> {
    // Embedding access key
    const embeddingAccessKeyField = this.page.locator('input[name="embeddingAccessKey"]');
    await embeddingAccessKeyField.waitFor({ state: 'visible' });
    await embeddingAccessKeyField.fill(accessKey);
    
    // Embedding secret key
    const embeddingSecretKeyField = this.page.locator('input[name="embeddingSecretKey"]');
    await embeddingSecretKeyField.fill(secretKey);
  }

  /**
   * Submit the connection form
   */
  async submitConnectionForm(): Promise<void> {
    const submitButton = this.page.locator('button[type="submit"]').filter({ hasText: /create connection|update connection/i });
    
    // Wait for form to be valid and button to be enabled
    await submitButton.waitFor({ state: 'visible' });
    
    // Wait for button to be enabled (with timeout)
    const maxAttempts = 10;
    for (let i = 0; i < maxAttempts; i++) {
      if (await submitButton.isEnabled()) {
        await submitButton.click();
        await this.page.waitForLoadState('domcontentloaded');
        return;
      }
      await this.page.waitForTimeout(500);
    }
    
    throw new Error('Submit button remained disabled after filling form');
  }

  /**
   * Verify connection creation success
   */
  async verifyConnectionSuccess(): Promise<void> {
    // Look for success dialog
    const successDialog = this.page.locator('[role="dialog"]').filter({ hasText: /connection succeeded|successfully configured/i });
    await successDialog.waitFor({ state: 'visible', timeout: 20000 });

    const viewConnectionsButton = successDialog.locator('button').filter({ hasText: /view.*connections/i });
    if (await viewConnectionsButton.isVisible()) {
      await viewConnectionsButton.click();
      await this.page.waitForLoadState('domcontentloaded');
    }
  }

  /**
   * Verify connection appears in the list
   */
  async verifyConnectionInList(connectionName: string): Promise<boolean> {
    // Navigate back to list to ensure we're looking at a fresh load
    await this.navigateToLLMConnections();

    // Cards use class "dataset-group-card"
    const connectionCard = this.page.locator('.dataset-group-card').filter({ hasText: connectionName });
    if (await connectionCard.count() > 0) {
      return connectionCard.first().isVisible();
    }

    // If not found on first page, check if there's a next page and search there
    const nextButton = this.page.locator('.pagination button, nav[aria-label*="pagination"] button')
      .filter({ hasText: /next|>/i }).first();

    while (await nextButton.isEnabled().catch(() => false)) {
      await nextButton.click();
      await this.page.waitForTimeout(800);
      const card = this.page.locator('.dataset-group-card').filter({ hasText: connectionName });
      if (await card.count() > 0) return card.first().isVisible();
    }

    return false;
  }
}

/**
 * Test data factory for LLM connections
 */
export class LLMConnectionTestData {
  static createAzureConnection(overrides: Partial<{
    connectionName: string;
    llmPlatform: string;
    llmModel: string;
    embeddingPlatform: string;
    embeddingModel: string;
    monthlyBudget: string;
    warnBudget: string;
    stopBudget: string;
    deploymentName: string;
    targetUri: string;
    apiKey: string;
    embeddingDeploymentName: string;
    embeddingTargetUri: string;
    embeddingApiKey: string;
    environment: 'testing' | 'production';
  }> = {}) {
    const defaultData = {
      connectionName: 'Test Azure OpenAI Connection',
      llmPlatform: 'Azure', 
      llmModel: 'GPT-4o',
      embeddingPlatform: 'Azure', 
      embeddingModel: 'text-embedding-3-large',
      monthlyBudget: '1000',
      warnBudget: '80',
      stopBudget: '95',
      deploymentName: 'test-gpt4o-deployment',
      targetUri: 'https://test-openai.openai.azure.com/',
      apiKey: 'sk-test-api-key-azure-12345',
      embeddingDeploymentName: 'test-embedding-deployment',
      embeddingTargetUri: 'https://test-openai.openai.azure.com/',
      embeddingApiKey: 'sk-test-embedding-api-key-azure-67890',
      environment: 'testing' as const,
    };

    return { ...defaultData, ...overrides };
  }

  static createAWSConnection(overrides: Partial<{
    connectionName: string;
    llmPlatform: string;
    llmModel: string;
    embeddingPlatform: string;
    embeddingModel: string;
    monthlyBudget: string;
    warnBudget: string;
    stopBudget: string;
    accessKey: string;
    secretKey: string;
    embeddingAccessKey: string;
    embeddingSecretKey: string;
    environment: 'testing' | 'production';
  }> = {}) {
    const defaultData = {
      connectionName: 'Test AWS Bedrock Connection',
      llmPlatform: 'AWS', 
      llmModel: 'Anthropic Claude 3.5 Sonnet',
      embeddingPlatform: 'AWS',
      embeddingModel: 'Amazon Titan Text Embeddings V2',
      monthlyBudget: '500',
      warnBudget: '75',
      stopBudget: '90',
      accessKey: 'AKIATEST12345',
      secretKey: 'test-secret-key-aws-67890',
      embeddingAccessKey: 'AKIATEST12345',
      embeddingSecretKey: 'test-secret-key-aws-67890',
      environment: 'testing' as const,
    };

    return { ...defaultData, ...overrides };
  }
}
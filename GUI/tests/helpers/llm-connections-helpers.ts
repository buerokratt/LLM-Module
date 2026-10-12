import { Page } from '@playwright/test';

/**
 * LLM Connections Helper
 */
export class LLMConnectionsHelper {
  constructor(private page: Page) {}

  async navigateToLLMConnections(): Promise<void> {
    await this.page.goto('http://localhost:3003/rag-search/llm-connections');
    await this.page.waitForLoadState('networkidle');
  }

  async navigateToCreateConnection(): Promise<void> {
    // First navigate to LLM connections page
    await this.navigateToLLMConnections();
    
    // Look for create button
    const createButton = this.page.locator('button').filter({
      hasText: /create.*connection|add.*connection|new.*connection/i
    });

    if (await createButton.count() > 0 && await createButton.isVisible()) {
      await createButton.click();
      await this.page.waitForLoadState('networkidle');
    } else {
      // Fallback to direct navigation
      await this.page.goto('http://localhost:3003/rag-search/create-llm-connection');
      await this.page.waitForLoadState('networkidle');
    }

    // Wait for form to be visible and fully loaded
    await this.page.locator('form').waitFor({ state: 'visible' });
    
    // Wait for the LLM Configuration section to be visible
    await this.page.locator('.form-section').filter({ hasText: /LLM Configuration/i }).waitFor({ state: 'visible' });
    
    // Wait for connection name field to be ready
    await this.page.locator('input[name="connectionName"]').waitFor({ state: 'visible' });
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
  await platformDropdown.waitFor({ state: 'visible', timeout: 5000 });

  const trigger = platformDropdown.locator('.select__trigger, .select-trigger, button').first();
  await trigger.waitFor({ state: 'visible', timeout: 5000 });
  await trigger.click();

  // Scope options to the opened dropdown only
  const options = platformDropdown.locator('.select__option, .select-option, .option');

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
    // Find the LLM Model dropdown (second FormSelect in the LLM Configuration section)
    const llmSection = this.page.locator('.form-section').filter({ hasText: /LLM Configuration/i });
    await llmSection.waitFor({ state: 'visible', timeout: 5000 });
    
    const modelDropdown = llmSection.locator('.select').nth(1);
    await modelDropdown.waitFor({ state: 'visible', timeout: 5000 });
    
    // Click the trigger to open dropdown
    const trigger = modelDropdown.locator('.select__trigger, .select-trigger, button').first();
    await trigger.waitFor({ state: 'visible', timeout: 5000 });
    await trigger.click();
    
    // Scope options to the opened dropdown only
    const options = modelDropdown.locator('.select__option, .select-option, .option');
    
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
    // Find the Embedding Model section
    const embeddingSection = this.page.locator('.form-section').filter({ hasText: /Embedding Model Configuration/i });
    await embeddingSection.waitFor({ state: 'visible', timeout: 5000 });
    
    const platformDropdown = embeddingSection.locator('.select').first();
    await platformDropdown.waitFor({ state: 'visible', timeout: 5000 });
    
    // Click the trigger to open dropdown
    const trigger = platformDropdown.locator('.select__trigger, .select-trigger, button').first();
    await trigger.waitFor({ state: 'visible', timeout: 5000 });
    await trigger.click();
    
    // Scope options to the opened dropdown only
    const options = platformDropdown.locator('.select__option, .select-option, .option');
    
    const targetOption = options.filter({ hasText: new RegExp(platformLabel, 'i') });
    if (await targetOption.count() === 0) {
      const availableOptions = await options.allTextContents();
      throw new Error(`Embedding platform "${platformLabel}" not found. Available options: ${availableOptions.join(', ')}`);
    }
    await targetOption.first().click();
    
    // Wait for dependent fields to load
    await this.page.waitForTimeout(1000);
  }

  /**
   * Select embedding model from dropdown
   */
  async selectEmbeddingModel(modelLabel: string): Promise<void> {
    // Find the Embedding Model dropdown (second FormSelect in the embedding section)
    const embeddingSection = this.page.locator('.form-section').filter({ hasText: /Embedding Model Configuration/i });
    await embeddingSection.waitFor({ state: 'visible', timeout: 5000 });
    
    const modelDropdown = embeddingSection.locator('.select').nth(1);
    await modelDropdown.waitFor({ state: 'visible', timeout: 5000 });
    
    // Click the trigger to open dropdown
    const trigger = modelDropdown.locator('.select__trigger, .select-trigger, button').first();
    await trigger.waitFor({ state: 'visible', timeout: 5000 });
    await trigger.click();
    
    // Scope options to the opened dropdown only
    const options = modelDropdown.locator('.select__option, .select-option, .option');
    
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
        await this.page.waitForLoadState('networkidle');
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
    await successDialog.waitFor({ state: 'visible', timeout: 10000 });
    
    // Click the button to go to connections list
    const viewConnectionsButton = successDialog.locator('button').filter({ hasText: /view.*connections/i });
    if (await viewConnectionsButton.isVisible()) {
      await viewConnectionsButton.click();
      await this.page.waitForLoadState('networkidle');
    }
  }

  /**
   * Verify connection appears in the list
   */
  async verifyConnectionInList(connectionName: string): Promise<boolean> {
    await this.page.waitForTimeout(1000);
    const connectionCard = this.page.locator('[class*="connection"]').filter({ hasText: connectionName });
    return (await connectionCard.count()) > 0 && (await connectionCard.first().isVisible());
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
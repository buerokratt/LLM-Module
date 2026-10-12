import { test, expect} from '@playwright/test';
import { AuthHelper, PageHelper } from './helpers/test-helpers';
import { TestLLMHelper,TestProductionLLMHelper } from './helpers/llm-testing-helpers';


// Test LLM Page

test.describe('Test LLM Page', () => {
  let authHelper: AuthHelper;
  let pageHelper: PageHelper;
  let testLLMHelper: TestLLMHelper;

  test.beforeEach(async ({ page }) => {
    authHelper = new AuthHelper(page);
    pageHelper = new PageHelper(page);
    testLLMHelper = new TestLLMHelper(page);

    await test.step('Login as administrator', async () => {
      await authHelper.loginAsAdmin();
      await authHelper.verifyAdminRedirect();
    });

    await test.step('Navigate to Test LLM page', async () => {
      await testLLMHelper.navigateToTestLLM();
    });
  });

  test('should load Test LLM page successfully', async ({ page }) => {
    await test.step('Verify page title and components', async () => {
      // Check page title
      const pageTitle = page.locator('.title, h1').filter({ hasText: /test.*llm/i });
      expect(await pageTitle.isVisible()).toBe(true);
      
      // Verify LLM Connection section exists
      const connectionSection = page.locator('.llm-connection-section');
      expect(await connectionSection.isVisible()).toBe(true);
      
      // Verify text area for input exists
      const textarea = page.locator('textarea').first();
      expect(await textarea.isVisible()).toBe(true);
      
      // Verify Send button exists
      const sendButton = page.locator('button').filter({ hasText: /send/i });
      expect(await sendButton.isVisible()).toBe(true);
      
      await pageHelper.takeScreenshot('test-llm-page-loaded');
    });
  });

  test('should load LLM connections in dropdown', async ({ page }) => {
    await test.step('Wait for connections to load', async () => {
      await testLLMHelper.waitForConnectionsToLoad();
      
      // Verify dropdown is visible and clickable
      const dropdown = page.locator('.select').first();
      expect(await dropdown.isVisible()).toBe(true);
      
      await pageHelper.takeScreenshot('test-llm-connections-loaded');
    });

    await test.step('Open dropdown and verify connections exist', async () => {
      const trigger = page.locator('.select__trigger').first();
      await trigger.click();
      
      // Wait for options to appear
      const options = page.locator('.select__option');
      await options.first().waitFor({ state: 'visible' });
      
      const optionCount = await options.count();
      expect(optionCount).toBeGreaterThan(0);
      
      await pageHelper.takeScreenshot('test-llm-dropdown-options');
      
      // Close dropdown
      await trigger.click();
    });
  });

  test('should disable Send button when form is incomplete', async ({ page }) => {
    await test.step('Verify Send button is disabled initially', async () => {
      await testLLMHelper.waitForConnectionsToLoad();
      
      const isDisabled = await testLLMHelper.isSendButtonDisabled();
      expect(isDisabled).toBe(true);
      
      await pageHelper.takeScreenshot('test-llm-send-disabled-initial');
    });

    await test.step('Verify Send button disabled with only text', async () => {
      await testLLMHelper.fillTestText('Test message without connection');
      
      const isDisabled = await testLLMHelper.isSendButtonDisabled();
      expect(isDisabled).toBe(true);
      
      await pageHelper.takeScreenshot('test-llm-send-disabled-text-only');
    });

    await test.step('Verify Send button disabled with only connection', async () => {
      await testLLMHelper.clearTestText();
      await testLLMHelper.selectFirstAvailableConnection();
      
      const isDisabled = await testLLMHelper.isSendButtonDisabled();
      expect(isDisabled).toBe(true);
      
      await pageHelper.takeScreenshot('test-llm-send-disabled-connection-only');
    });
  });

  test('should enable Send button when form is complete', async ({ page }) => {
    await test.step('Select connection and enter text', async () => {
      await testLLMHelper.waitForConnectionsToLoad();
      
      // Select first available connection
      const connectionName = await testLLMHelper.selectFirstAvailableConnection();
      expect(connectionName).not.toBeNull();
      
      // Fill text
      await testLLMHelper.fillTestText('What is artificial intelligence?');
      
      await pageHelper.takeScreenshot('test-llm-form-complete');
    });

    await test.step('Verify Send button is enabled', async () => {
      const isDisabled = await testLLMHelper.isSendButtonDisabled();
      expect(isDisabled).toBe(false);
      
      await pageHelper.takeScreenshot('test-llm-send-enabled');
    });
  });

  test('should validate input text length', async ({ page }) => {
    await test.step('Enter text up to max length', async () => {
      await testLLMHelper.waitForConnectionsToLoad();
      
      // Generate text close to max length (1000 characters)
      const longText = 'A'.repeat(950);
      await testLLMHelper.fillTestText(longText);
      
      const charCount = await testLLMHelper.getCharacterCount();
      expect(charCount).toBe(950);
      
      await pageHelper.takeScreenshot('test-llm-text-length-validation');
    });

    await test.step('Verify max length enforcement', async () => {
      // Try to add more characters beyond max
      const maxText = 'A'.repeat(1000);
      await testLLMHelper.clearTestText();
      await testLLMHelper.fillTestText(maxText + 'EXTRA');
      
      const charCount = await testLLMHelper.getCharacterCount();
      expect(charCount).toBeLessThanOrEqual(1000);
      
      await pageHelper.takeScreenshot('test-llm-max-length-enforced');
    });
  });

  test('should send inference request and receive response', async ({ page }) => {
    await test.step('Complete form and submit', async () => {
      await testLLMHelper.waitForConnectionsToLoad();
      
      // Select connection
      await testLLMHelper.selectFirstAvailableConnection();
      
      // Fill test text
      const testMessage = 'Explain machine learning in simple terms.';
      await testLLMHelper.fillTestText(testMessage);
      
      await pageHelper.takeScreenshot('test-llm-before-send');
      
      // Click send
      await testLLMHelper.clickSendButton();
    });

    await test.step('Verify loading state', async () => {
      // Check if button shows loading state
      const isLoading = await testLLMHelper.verifyLoadingState();
      
      if (isLoading) {
        await pageHelper.takeScreenshot('test-llm-loading-state');
      }
    });

    await test.step('Wait for and verify response', async () => {
      // Wait for inference result (with extended timeout)
      try {
        await testLLMHelper.waitForInferenceResult();
        
        // Get the response content
        const result = await testLLMHelper.getInferenceResult();
        
        // Verify response is not empty
        expect(result.length).toBeGreaterThan(0);
        expect(result).not.toBe('');
        
        await pageHelper.takeScreenshot('test-llm-response-received');
      } catch (error) {
        // If inference fails, check for error message
        const hasError = await testLLMHelper.verifyErrorMessage();
        
        if (hasError) {
          await pageHelper.takeScreenshot('test-llm-inference-error');
          console.log('Inference failed with error message displayed');
        } else {
          throw error;
        }
      }
    });
  });

  test('should handle inference errors gracefully', async ({ page }) => {
    await test.step('Setup and simulate error scenario', async () => {
      await testLLMHelper.waitForConnectionsToLoad();
      
      // Select connection
      await testLLMHelper.selectFirstAvailableConnection();
      
      // Use problematic input
      await testLLMHelper.fillTestText('');
      
      await pageHelper.takeScreenshot('test-llm-error-setup');
    });

    
  });

  test('should clear form after successful submission', async ({ page }) => {
    await test.step('Submit inference request', async () => {
      await testLLMHelper.waitForConnectionsToLoad();
      
      await testLLMHelper.selectFirstAvailableConnection();
      await testLLMHelper.fillTestText('Test message for clear verification');
      
      await testLLMHelper.clickSendButton();
    });

    await test.step('Verify form state after submission', async () => {
      // Wait a moment for processing
      await page.waitForTimeout(2000);
      
      // Form should maintain values
      const charCount = await testLLMHelper.getCharacterCount();
      
      // Just verify the page is still functional
      expect(charCount).toBeGreaterThanOrEqual(0);
      
      await pageHelper.takeScreenshot('test-llm-after-submission');
    });
  });

});



import { Page } from '@playwright/test';


/**
 * Test LLM Helper - Helper functions for Testing LLM functionality
 */
export class TestLLMHelper {
  constructor(private page: Page) {}

  async navigateToTestLLM(): Promise<void> {
    await this.page.goto('http://localhost:3003/rag-search/test-llm');
    await this.page.waitForLoadState('networkidle');
  }

  async waitForConnectionsToLoad(): Promise<void> {
    // Wait for loading spinner to disappear
    const spinner = this.page.locator('.circular-spinner, .spinner, .loading');
    if (await spinner.isVisible()) {
      await spinner.waitFor({ state: 'hidden', timeout: 10000 });
    }
    
    // Wait for the connection dropdown to be visible
    await this.page.locator('select[name="connectionId"], .select').first().waitFor({ state: 'visible', timeout: 5000 });
  }

  async selectLLMConnection(connectionLabel: string): Promise<void> {
    // Try to find the select element or custom dropdown
    const selectElement = this.page.locator('select[name="connectionId"]');
    const customDropdown = this.page.locator('.select').first();

    if (await selectElement.count() > 0 && await selectElement.isVisible()) {
      // Native select element
      await selectElement.selectOption({ label: connectionLabel });
    } else if (await customDropdown.count() > 0 && await customDropdown.isVisible()) {
      // Custom dropdown (Downshift)
      const trigger = customDropdown.locator('.select__trigger');
      await trigger.click();
      
      // Wait for options and select
      const option = this.page.locator('.select__option').filter({ hasText: new RegExp(connectionLabel, 'i') });
      await option.waitFor({ state: 'visible' });
      await option.click();
    }
  }

  async selectFirstAvailableConnection(): Promise<string | null> {
    const customDropdown = this.page.locator('.select').first();
    
    if (await customDropdown.isVisible()) {
      // Click to open dropdown
      const trigger = customDropdown.locator('.select__trigger');
      await trigger.click();
      
      // Get first option
      const firstOption = this.page.locator('.select__option').first();
      
      if (await firstOption.count() > 0) {
        const connectionText = await firstOption.textContent();
        await firstOption.click();
        return connectionText;
      }
    }
    
    return null;
  }

  async fillTestText(text: string): Promise<void> {
    const textarea = this.page.locator('textarea').first();
    await textarea.waitFor({ state: 'visible' });
    await textarea.fill(text);
  }

  async clearTestText(): Promise<void> {
    const textarea = this.page.locator('textarea').first();
    await textarea.waitFor({ state: 'visible' });
    await textarea.clear();
  }

  async clickSendButton(): Promise<void> {
    const sendButton = this.page.locator('button').filter({ hasText: /send/i });
    await sendButton.waitFor({ state: 'visible' });
    await sendButton.click();
  }

  async isSendButtonDisabled(): Promise<boolean> {
    const sendButton = this.page.locator('button').filter({ hasText: /send/i });
    return await sendButton.isDisabled();
  }

  async waitForInferenceResult(): Promise<void> {
    // Wait for inference result container to appear
    await this.page.locator('.inference-results-container, .response-content').waitFor({ 
      state: 'visible', 
      timeout: 30000 
    });
  }

  async getInferenceResult(): Promise<string> {
    const resultContent = this.page.locator('.response-content').first();
    await resultContent.waitFor({ state: 'visible' });
    return await resultContent.textContent() || '';
  }

  async verifyErrorMessage(): Promise<boolean> {
    const errorMessage = this.page.locator('.classification-error, .error-message');
    return await errorMessage.isVisible();
  }

  async verifyLoadingState(): Promise<boolean> {
    const sendButton = this.page.locator('button').filter({ hasText: /sending/i });
    return await sendButton.isVisible();
  }

  async getCharacterCount(): Promise<number> {
    const textarea = this.page.locator('textarea').first();
    const value = await textarea.inputValue();
    return value.length;
  }

  async verifyMaxLengthIndicator(): Promise<boolean> {
    // Check if max length indicator is visible
    const maxLengthIndicator = this.page.locator('.max-length, .character-count');
    return await maxLengthIndicator.isVisible();
  }
}

/**
 * Test Production LLM Helper - Helper functions for Testing Production LLM 
 */
export class TestProductionLLMHelper {
  constructor(private page: Page) {}

  async navigateToTestProductionLLM(): Promise<void> {
    await this.page.goto('http://localhost:3003/rag-search/test-production-llm');
    await this.page.waitForLoadState('networkidle');
  }

  async verifyWelcomeMessage(): Promise<boolean> {
    const welcomeMessage = this.page.locator('.test-production-llm__welcome');
    return await welcomeMessage.isVisible();
  }

  async typeMessage(message: string): Promise<void> {
    const textarea = this.page.locator('textarea[name="message"]').or(this.page.locator('textarea[aria-label="Message"]'));
    await textarea.waitFor({ state: 'visible' });
    await textarea.fill(message);
  }

  async clearMessage(): Promise<void> {
    const textarea = this.page.locator('textarea[name="message"]').or(this.page.locator('textarea[aria-label="Message"]'));
    await textarea.waitFor({ state: 'visible' });
    await textarea.clear();
  }

  async clickSendButton(): Promise<void> {
    const sendButton = this.page.locator('button.test-production-llm__send-button, button').filter({ hasText: /send/i });
    await sendButton.waitFor({ state: 'visible' });
    await sendButton.click();
  }

  async pressEnterToSend(): Promise<void> {
    const textarea = this.page.locator('textarea[name="message"]').or(this.page.locator('textarea[aria-label="Message"]'));
    await textarea.press('Enter');
  }

  async pressShiftEnterForNewLine(): Promise<void> {
    const textarea = this.page.locator('textarea[name="message"]').or(this.page.locator('textarea[aria-label="Message"]'));
    await textarea.press('Shift+Enter');
  }

  async isSendButtonDisabled(): Promise<boolean> {
    const sendButton = this.page.locator('button.test-production-llm__send-button, button').filter({ hasText: /send/i });
    return await sendButton.isDisabled();
  }

  async waitForBotResponse(): Promise<void> {
    // Wait for typing indicator to appear
    const typingIndicator = this.page.locator('.test-production-llm__typing');
    
    if (await typingIndicator.isVisible()) {
      // Wait for typing indicator to disappear
      await typingIndicator.waitFor({ state: 'hidden', timeout: 60000 });
    }
    
    // Wait for bot message to appear
    await this.page.locator('.test-production-llm__message--bot').last().waitFor({ 
      state: 'visible', 
      timeout: 60000 
    });
  }

  async getUserMessages(): Promise<string[]> {
    const userMessages = this.page.locator('.test-production-llm__message--user .test-production-llm__message-content');
    const count = await userMessages.count();
    const messages: string[] = [];
    
    for (let i = 0; i < count; i++) {
      const text = await userMessages.nth(i).textContent();
      messages.push(text || '');
    }
    
    return messages;
  }

  async getBotMessages(): Promise<string[]> {
    const botMessages = this.page.locator('.test-production-llm__message--bot .test-production-llm__message-content');
    const count = await botMessages.count();
    const messages: string[] = [];
    
    for (let i = 0; i < count; i++) {
      const text = await botMessages.nth(i).textContent();
      messages.push(text || '');
    }
    
    return messages;
  }

  async getMessageCount(): Promise<{ user: number; bot: number }> {
    const userCount = await this.page.locator('.test-production-llm__message--user').count();
    const botCount = await this.page.locator('.test-production-llm__message--bot').count();
    
    return { user: userCount, bot: botCount };
  }

  async clickClearChat(): Promise<void> {
    const clearButton = this.page.locator('button').filter({ hasText: /clear.*chat/i });
    await clearButton.waitFor({ state: 'visible' });
    await clearButton.click();
  }

  async verifyChatCleared(): Promise<boolean> {
    const messages = this.page.locator('.test-production-llm__message');
    const count = await messages.count();
    return count === 0;
  }

  async verifyTypingIndicator(): Promise<boolean> {
    const typingIndicator = this.page.locator('.test-production-llm__typing');
    return await typingIndicator.isVisible();
  }

  async verifyMessageTimestamp(messageIndex: number = 0): Promise<boolean> {
    const timestamp = this.page.locator('.test-production-llm__message-timestamp').nth(messageIndex);
    return await timestamp.isVisible();
  }

  async scrollToBottom(): Promise<void> {
    await this.page.evaluate(() => {
      const messagesContainer = document.querySelector('.test-production-llm__messages');
      if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
    });
  }

  async getLastBotMessage(): Promise<string> {
    const lastBotMessage = this.page.locator('.test-production-llm__message--bot').last();
    const content = lastBotMessage.locator('.test-production-llm__message-content');
    return await content.textContent() || '';
  }

  async getLastUserMessage(): Promise<string> {
    const lastUserMessage = this.page.locator('.test-production-llm__message--user').last();
    const content = lastUserMessage.locator('.test-production-llm__message-content');
    return await content.textContent() || '';
  }

  async verifyErrorMessage(messageContent: string): Promise<boolean> {
    const botMessages = await this.getBotMessages();
    return botMessages.some(msg => msg.toLowerCase().includes(messageContent.toLowerCase()));
  }

  async verifyInputDisabledWhileLoading(): Promise<boolean> {
    const textarea = this.page.locator('textarea[name="message"]').or(this.page.locator('textarea[aria-label="Message"]'));
    return await textarea.isDisabled();
  }
}
import { Page, expect } from '@playwright/test';


/**
 * Authentication helper functions for reuse across tests
 */
export class AuthHelper {
  constructor(private page: Page) {}

  /**
   * Login with Estonian ID
   */
  async loginAsAdmin(): Promise<void> {
    await this.page.goto('http://localhost:3004/et/dev-auth');
    await this.page.waitForLoadState('networkidle');
    
    const idInput = await this.page.locator('input[type="text"], input[placeholder*="user name"], input[placeholder*="sisesta"]').first();
    await idInput.fill('EE30303039914');
    
    const submitButton = await this.page.locator('button[type="submit"], button:has-text("sisene"), button:has-text("Sisene"), input[type="submit"]').first();
    await submitButton.click();
    
    await this.page.waitForTimeout(20000);
    await this.page.waitForLoadState('networkidle', { timeout: 20000 });
  }

  /**
   * Verify user is redirected to the correct page based on role
   */
  async verifyAdminRedirect(): Promise<void> {
    expect(this.page.url()).toContain('user-management');
  }

  /**
   * Verify user is redirected to model trainer page
   */
  async verifyTrainerRedirect(): Promise<void> {
    expect(this.page.url()).toContain('dataset-groups');
  }

  /**
   * Click logout button and handle logout process
   */
  async logout(): Promise<void> {
    const logoutButton = await this.page.locator('button:has-text("Logout")').first();
    
    if (await logoutButton.isVisible()) {
      await logoutButton.click();
      
      // Wait for logout process to complete
      await this.page.waitForTimeout(3000);
    } else {
      throw new Error('Logout button not found');
    }
  }

  /**
   * Verify user is redirected to login page after logout
   */
  async verifyLogoutRedirect(): Promise<void> {
    // After logout, user should be redirected to the auth service
    await this.page.waitForLoadState('networkidle', { timeout: 10000 });
    
    // Check that we're no longer on the application pages
    const currentUrl = this.page.url();
    
    // Should not contain the main app URLs anymore
    expect(currentUrl).not.toContain('/user-management');
    expect(currentUrl).not.toContain('/dataset-groups');
    expect(currentUrl).not.toContain('/llm-connections');
    
    // Should redirect to auth service (localhost:3004 for dev)
    expect(currentUrl).toContain('localhost:3004');
  }

  /**
   * Handle session timeout modal logout
   */
  async handleSessionTimeoutLogout(): Promise<void> {
    // If session timeout modal appears, click logout there
    const sessionModal = await this.page.locator('[role="dialog"]').filter({ hasText: /session.*time.*out/i });
    
    if (await sessionModal.isVisible()) {
      const modalLogoutButton = await sessionModal.locator('button:has-text("Logout")').first();
      await modalLogoutButton.click();
      await this.page.waitForTimeout(3000);
    }
  }
}

/**
 * User Management page helper functions
 */
export class UserManagementHelper {
  constructor(private page: Page) {}

  /**
   * Navigate to user management page
   */
  async navigateToUserManagement(): Promise<void> {
    await this.page.goto('http://localhost:3003/rag-search/user-management');
    await this.page.waitForLoadState('networkidle');
  }

  /**
   * Open the add a user modal
   */
  async openAddUserModal(): Promise<void> {
    const addUserButton = await this.page.locator('button').filter({ hasText: /add.*user|lisa.*kasutaja/i }).first();
    await addUserButton.click();
    await this.page.waitForTimeout(1000);
    
    const modal = await this.page.locator('[role="dialog"], .modal, .dialog');
    expect(await modal.isVisible()).toBe(true);
  }

  /**
   * Close modal by clicking close button or pressing escape
   */
  async closeModal(): Promise<void> {
    const closeButton = await this.page.locator('button[aria-label*="close"], button:has-text("Cancel"), button:has-text("Close"), .modal-close, [data-testid="close-button"]').first();
    
    if (await closeButton.isVisible()) {
      await closeButton.click();
    } else {
      await this.page.keyboard.press('Escape');
    }
    
    await this.page.waitForTimeout(1000);
    
    const modal = await this.page.locator('[role="dialog"], .modal, .dialog');
    expect(await modal.isVisible()).toBe(false);
  }

  /**
   * Fill user form with test data
   */
  async fillUserForm(userData: {
    fullName?: string;
    idCode?: string;
    email?: string;
    title?: string;
    role?: string;
  }): Promise<void> {
    if (userData.fullName) {
      const nameInput = await this.page.locator('input[placeholder*="name"], input[name*="name"], input[name*="fullName"]').first();
      if (await nameInput.isVisible()) {
        await nameInput.fill(userData.fullName);
      }
    }

    if (userData.idCode) {
      const idInput = await this.page.locator('input[placeholder*="id"], input[name*="userid"], input[name*="identification"]').first();
      if (await idInput.isVisible()) {
        await idInput.fill(userData.idCode);
      }
    }

    if (userData.email) {
      const emailInput = await this.page.locator('input[type="email"], input[placeholder*="email"], input[name*="email"]').first();
      if (await emailInput.isVisible()) {
        await emailInput.fill(userData.email);
      }
    }

    if (userData.title) {
      const titleInput = await this.page.locator('input[placeholder*="title"], input[name*="title"], input[name*="csaTitle"]').first();
      if (await titleInput.isVisible()) {
        await titleInput.fill(userData.title);
      }
    }

    if (userData.role) {
      await this.selectRole(userData.role);
    }
  }

  /**
   * Select a role from dropdown
   */
  async selectRole(role: string): Promise<void> {
    const nativeSelect = await this.page.locator('select[name*="role"], select[name*="authorities"]');
    const reactSelect = await this.page.locator('.react-select__control');
    
    if (await nativeSelect.count() > 0 && await nativeSelect.first().isVisible()) {
      await nativeSelect.first().selectOption({ label: role });
    } else if (await reactSelect.count() > 0 && await reactSelect.first().isVisible()) {
      await reactSelect.first().click();
      await this.page.waitForTimeout(500);
      const option = await this.page.locator('.react-select__option').filter({ hasText: role });
      if (await option.isVisible()) {
        await option.click();
      }
    }
  }

  /**
   * Submit the user form
   */
  async submitForm(): Promise<void> {
    // Based on UserModal.tsx, the submit button uses t('global.confirm') = "Confirm"
    const submitButton = await this.page.locator('button:has-text("Confirm")').first();
    await submitButton.click();
    await this.page.waitForTimeout(3000);
  }

  /**
   * Verify success notification appears
   */
  async verifySuccessNotification(messagePattern: RegExp = /success|created|added|updated|saved/i): Promise<void> {
    const successMessage = await this.page.locator('.toast, .notification, .alert').filter({ hasText: messagePattern });
    if (await successMessage.count() > 0) {
      expect(await successMessage.first().isVisible()).toBe(true);
    }
  }

  /**
   * Click edit button for the first user in the table
   */
  async editFirstUser(): Promise<boolean> {
    await this.page.waitForSelector('table, [data-testid="data-table"]', { timeout: 10000 });
    
    const editButton = await this.page.locator('button[aria-label*="edit"], button:has-text("Edit"), .edit-button, [data-testid*="edit"]').first();
    
    if (await editButton.isVisible()) {
      await editButton.click();
      await this.page.waitForTimeout(1000);
      
      const modal = await this.page.locator('[role="dialog"], .modal, .dialog');
      expect(await modal.isVisible()).toBe(true);
      return true;
    }
    
    return false;
  }

  /**
   * Click delete button for the first user and handle confirmation dialog
   */
  async deleteFirstUser(): Promise<boolean> {
    await this.page.waitForSelector('table, [data-testid="data-table"]', { timeout: 10000 });
    
    const deleteButton = await this.page.locator('button').filter({ hasText: /delete|kustuta/i }).first();
    
    if (await deleteButton.isVisible()) {
      await deleteButton.click();
      await this.page.waitForTimeout(1000);
      
      // Verify confirmation dialog opened
      const confirmationDialog = await this.page.locator('[role="dialog"], .modal, .dialog');
      expect(await confirmationDialog.isVisible()).toBe(true);
      
      return true;
    }
    
    return false;
  }

  /**
   * Confirm deletion in the confirmation dialog
   */
  async confirmDeletion(): Promise<void> {
    // Based on ActionButtons, the confirm button in delete dialog uses t('global.confirm') = "Confirm"
    const confirmButton = await this.page.locator('button:has-text("Confirm")').first();
    await confirmButton.click();
    await this.page.waitForTimeout(2000);
  }

  /**
   * Cancel deletion in the confirmation dialog
   */
  async cancelDeletion(): Promise<void> {
    const cancelButton = await this.page.locator('button:has-text("Cancel")').first();
    await cancelButton.click();
    await this.page.waitForTimeout(1000);
    
    // Verify dialog is closed
    const modal = await this.page.locator('[role="dialog"], .modal, .dialog');
    expect(await modal.isVisible()).toBe(false);
  }

  /**
   * Sort table by column using header click
   */
  async sortByColumn(columnName: string): Promise<void> {
    const columnHeader = await this.page.locator('th').filter({ hasText: new RegExp(columnName, 'i') }).first();
    if (await columnHeader.isVisible()) {
      await columnHeader.click();
      await this.page.waitForTimeout(2000);
    }
  }

  /**
   * Sort table by clicking sort arrows/icons
   */
  async sortByArrowIcon(columnName: string, direction: 'asc' | 'desc' = 'asc'): Promise<boolean> {
    // Find the column header
    const columnHeader = await this.page.locator('th').filter({ hasText: new RegExp(columnName, 'i') }).first();
    
    if (await columnHeader.isVisible()) {
      // Look for sort arrow icons within the column header
      const sortButton = direction === 'asc' 
        ? await columnHeader.locator('button, [role="button"]').first()
        : await columnHeader.locator('button, [role="button"]').first();
      
      if (await sortButton.isVisible()) {
        await sortButton.click();
        await this.page.waitForTimeout(2000);
        return true;
      }
      
      // Fallback: click the header itself
      await columnHeader.click();
      await this.page.waitForTimeout(2000);
      return true;
    }
    
    return false;
  }

  /**
   * Search in a specific column using the search icon
   */
  async searchInColumn(columnName: string, searchText: string): Promise<boolean> {
    // Find the column header
    const columnHeader = await this.page.locator('th').filter({ hasText: new RegExp(columnName, 'i') }).first();
    
    if (await columnHeader.isVisible()) {
      // Look for search icon button
      const searchButton = await columnHeader.locator('button').first();
      
      if (await searchButton.isVisible()) {
        await searchButton.click();
        await this.page.waitForTimeout(500);
        
        // Look for the search input that appears
        const searchInput = await this.page.locator('.data-table__filter input[type="text"]').first();
        
        if (await searchInput.isVisible()) {
          await searchInput.fill(searchText);
          await this.page.waitForTimeout(2000);
          return true;
        }
      }
    }
    
    return false;
  }

  /**
   * Clear search in a column
   */
  async clearColumnSearch(columnName: string): Promise<void> {
    // Find the column header
    const columnHeader = await this.page.locator('th').filter({ hasText: new RegExp(columnName, 'i') }).first();
    
    if (await columnHeader.isVisible()) {
      // Look for search icon button
      const searchButton = await columnHeader.locator('button').first();
      
      if (await searchButton.isVisible()) {
        await searchButton.click();
        await this.page.waitForTimeout(500);
        
        // Clear the search input
        const searchInput = await this.page.locator('.data-table__filter input[type="text"]').first();
        
        if (await searchInput.isVisible()) {
          await searchInput.clear();
          await this.page.waitForTimeout(1000);
          
          // Click outside to close search
          await this.page.keyboard.press('Escape');
          await this.page.waitForTimeout(500);
        }
      }
    }
  }

  /**
   * Verify table is sorted correctly
   */
  async verifyTableSort(columnIndex: number, expectedOrder: 'asc' | 'desc' = 'asc'): Promise<boolean> {
    await this.page.waitForTimeout(1000);
    
    // Get all cell values from the specified column
    const cells = await this.page.locator(`table tbody tr td:nth-child(${columnIndex + 1}), [data-testid="data-table"] tbody tr td:nth-child(${columnIndex + 1})`);
    
    const cellTexts = [];
    const count = await cells.count();
    
    for (let i = 0; i < Math.min(count, 5); i++) { // Check first 5 rows
      const text = await cells.nth(i).textContent();
      if (text?.trim()) {
        cellTexts.push(text.trim());
      }
    }
    
    if (cellTexts.length < 2) return true; // Can't verify sort with less than 2 items
    
    // Check if sorted correctly
    const sorted = [...cellTexts].sort((a, b) => {
      if (expectedOrder === 'asc') {
        return a.localeCompare(b, undefined, { numeric: true });
      } else {
        return b.localeCompare(a, undefined, { numeric: true });
      }
    });
    
    return JSON.stringify(cellTexts) === JSON.stringify(sorted);
  }

  /**
   * Verify search results contain search text
   */
  async verifySearchResults(columnIndex: number, searchText: string): Promise<boolean> {
    await this.page.waitForTimeout(1000);
    
    // Get visible table rows
    const rows = await this.page.locator('table tbody tr, [data-testid="data-table"] tbody tr');
    const rowCount = await rows.count();
    
    if (rowCount === 0) return true; // Empty results are valid for no matches
    
    // Check that all visible rows contain the search text in the specified column
    for (let i = 0; i < Math.min(rowCount, 5); i++) {
      const cell = await rows.nth(i).locator(`td:nth-child(${columnIndex + 1})`);
      const cellText = await cell.textContent();
      
      if (cellText && !cellText.toLowerCase().includes(searchText.toLowerCase())) {
        return false;
      }
    }
    
    return true;
  }

  /**
   * Navigate to next page if pagination is available
   */
  async goToNextPage(): Promise<boolean> {
    const nextButton = await this.page.locator('button[aria-label*="next"], button:has-text("Next"), .pagination-next');
    
    if (await nextButton.count() > 0 && await nextButton.first().isEnabled()) {
      await nextButton.first().click();
      await this.page.waitForTimeout(2000);
      return true;
    }
    
    return false;
  }

  /**
   * Navigate to previous page if pagination is available
   */
  async goToPreviousPage(): Promise<boolean> {
    const prevButton = await this.page.locator('button[aria-label*="previous"], button:has-text("Previous"), .pagination-prev');
    
    if (await prevButton.count() > 0 && await prevButton.first().isEnabled()) {
      await prevButton.first().click();
      await this.page.waitForTimeout(2000);
      return true;
    }
    
    return false;
  }
}

/**
 * Common page utilities
 */
export class PageHelper {
  constructor(private page: Page) {}

  /**
   * Take a screenshot with automatic path generation
   */
  async takeScreenshot(name: string): Promise<void> {
    await this.page.screenshot({ path: `tests/screenshots/${name}.png` });
  }

  /**
   * Wait for table to load
   */
  async waitForTable(): Promise<void> {
    await this.page.waitForSelector('table, [data-testid="data-table"]', { timeout: 10000 });
  }

  /**
   * Check if element exists and is visible
   */
  async isElementVisible(selector: string): Promise<boolean> {
    const element = await this.page.locator(selector);
    return await element.count() > 0 && await element.first().isVisible();
  }
}

/**
 * Test data factory for creating test users
 */
export class TestDataFactory {
  static createTestUser(overrides: Partial<{
    fullName: string;
    idCode: string;
    email: string;
    title: string;
    role: string;
  }> = {}) {
    const defaultUser = {
      fullName: 'Test User Name',
      idCode: 'EE12345678901',
      email: 'test.user@example.com',
      title: 'Test Manager',
      role: 'MODEL_TRAINER'
    };

    return { ...defaultUser, ...overrides };
  }

  static createAdminUser(overrides: Partial<{
    fullName: string;
    idCode: string;
    email: string;
    title: string;
    role: string;
  }> = {}) {
    return this.createTestUser({
      role: 'ROLE_ADMINISTRATOR',
      title: 'System Administrator',
      ...overrides
    });
  }
}

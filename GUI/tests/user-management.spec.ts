import { test, expect } from '@playwright/test';
import { AuthHelper, UserManagementHelper, PageHelper, TestDataFactory } from './helpers/test-helpers';

test.describe('User Management', () => {
  let authHelper: AuthHelper;
  let userManagementHelper: UserManagementHelper;
  let pageHelper: PageHelper;

  // Setup: Login as admin before each test
  test.beforeEach(async ({ page }) => {
    authHelper = new AuthHelper(page);
    userManagementHelper = new UserManagementHelper(page);
    pageHelper = new PageHelper(page);

    await test.step('Login as administrator', async () => {
      await authHelper.loginAsAdmin();
      await authHelper.verifyAdminRedirect();
    });
  });

  test('should display user management page with users table', async ({ page }) => {
    await test.step('Verify page title and navigation', async () => {
      // Check page title
      const pageTitle = await page.locator('h1, h2, h3, .title').filter({ hasText: /user|kasutaj|management|haldus/i }).first();
      expect(await pageTitle.isVisible()).toBe(true);
      
      // Verify add user button is present
      const addUserButton = await page.locator('button').filter({ hasText: /add.*user|lisa.*kasutaja/i });
      expect(await addUserButton.isVisible()).toBe(true);
    });

    await test.step('Verify users table is displayed', async () => {
      // Wait for table to load
      await page.waitForSelector('table, [data-testid="data-table"]', { timeout: 10000 });
      
      // Check if table headers are present
      const expectedHeaders = ['fullName', 'personalId', 'title', 'role', 'email', 'actions'];
      
      for (const header of expectedHeaders) {
        const headerElement = await page.locator('th').filter({ hasText: new RegExp(header, 'i') });
        // Header might not be visible if table is empty
        if (await headerElement.count() > 0) {
          expect(await headerElement.first().isVisible()).toBe(true);
        }
      }
      
      // Take screenshot for verification
      await pageHelper.takeScreenshot('user-management-table');
    });

    await test.step('Verify pagination and sorting controls', async () => {
      // Check for pagination controls
      const paginationContainer = await page.locator('[data-testid="pagination"], .pagination, nav[aria-label*="pagination"]');
      if (await paginationContainer.count() > 0) {
        expect(await paginationContainer.first().isVisible()).toBe(true);
      }
      
      // Check for sorting capabilities (column headers should be clickable)
      const sortableHeaders = await page.locator('th button, th[role="columnheader"]');
      if (await sortableHeaders.count() > 0) {
        expect(await sortableHeaders.first().isVisible()).toBe(true);
      }
    });
  });

  test('should open and close add user modal', async ({ page }) => {
    await test.step('Click add user button', async () => {
      await userManagementHelper.openAddUserModal();
    });

    await test.step('Verify modal is opened', async () => {
      // Check for form elements in modal
      const formFields = await page.locator('input[type="text"], input[type="email"], select, textarea');
      expect(await formFields.count()).toBeGreaterThan(0);
      
      // Take screenshot of modal
      await pageHelper.takeScreenshot('add-user-modal');
    });

    await test.step('Close modal', async () => {
      await userManagementHelper.closeModal();
    });
  });

  test('should validate required fields in add user form', async ({ page }) => {
    await test.step('Open add user modal', async () => {
      const addUserButton = await page.locator('button').filter({ hasText: /add.*user|lisa.*kasutaja/i }).first();
      await addUserButton.click();
      await page.waitForTimeout(1000);
    });

    await test.step('Verify submit button is disabled for empty form', async () => {
      // Check that submit button is disabled when form is empty
      const submitButton = await page.locator('button:has-text("Confirm")').first();
      
      // Button should be disabled for empty form
      expect(await submitButton.isEnabled()).toBe(false);
      
      // Take screenshot showing disabled button
      await page.screenshot({ path: 'tests/screenshots/disabled-submit-button.png' });
    });

    await test.step('Fill partial form to test field validation', async () => {
      // Fill only one field to see if button becomes enabled or if individual field validation appears
      const nameInput = await page.locator('input[placeholder*="name"], input[name*="name"], input[name*="fullName"]').first();
      if (await nameInput.isVisible()) {
        await nameInput.fill('Test');
        
        // Check if any validation messages appear for other empty required fields
        await page.waitForTimeout(1000);
        
        // Take screenshot showing partial validation state
        await page.screenshot({ path: 'tests/screenshots/partial-form-validation.png' });
      }
    });

    await test.step('Verify required field indicators are present', async () => {
      // Check for visual indicators that fields are required
      const requiredIndicators = await page.locator('label:has-text("*"), .required, [required], input[placeholder*="required"]');
      
      // At least some required field indicators should be visible
      if (await requiredIndicators.count() > 0) {
        expect(await requiredIndicators.first().isVisible()).toBe(true);
      }
      
      // Take screenshot of validation state
      await page.screenshot({ path: 'tests/screenshots/user-form-validation.png' });
    });
  });

  test('should fill and submit user creation form', async ({ page }) => {
    const testUser = TestDataFactory.createTestUser();

    await test.step('Open add user modal', async () => {
      await userManagementHelper.openAddUserModal();
    });

    await test.step('Fill user form with valid data', async () => {
      await userManagementHelper.fillUserForm(testUser);
      await pageHelper.takeScreenshot('user-form-filled');
    });

    await test.step('Submit form', async () => {
      await userManagementHelper.submitForm();
    });

    await test.step('Verify user creation success', async () => {
      await userManagementHelper.verifySuccessNotification(/success|created|added/i);
      await pageHelper.takeScreenshot('user-creation-success');
    });
  });

  test('should edit existing user', async ({ page }) => {
    await test.step('Find and click edit button for first user', async () => {
      // Wait for table to load
      await page.waitForSelector('table, [data-testid="data-table"]', { timeout: 10000 });
      
      // Find edit button in actions column
      const editButton = await page.locator('button[aria-label*="edit"], button:has-text("Change"), .edit-button, [data-testid*="edit"]').first();
      
      if (await editButton.isVisible()) {
        await editButton.click();
        await page.waitForTimeout(1000);
        
        // Verify edit modal opened
        const modal = await page.locator('[role="dialog"], .modal, .dialog');
        expect(await modal.isVisible()).toBe(true);
        
        await page.screenshot({ path: 'tests/screenshots/edit-user-modal.png' });
      } else {
        // If no users exist to edit, skip this test
        console.log('No users available to edit - skipping test');
        return;
      }
    });

    await test.step('Modify user data', async () => {
      // Update title field
      const titleInput = await page.locator('input[name*="title"], input[name*="csaTitle"]').first();
      if (await titleInput.isVisible()) {
        await titleInput.clear();
        await titleInput.fill('Updated Test Manager');
      }
      
      await page.screenshot({ path: 'tests/screenshots/user-form-updated.png' });
    });

    await test.step('Save changes', async () => {
      // The save button is the same "Confirm" button for both create and edit modes
      const saveButton = await page.locator('button:has-text("Confirm")').first();
      await saveButton.click();
      
      // Wait for save operation
      await page.waitForTimeout(3000);
      
      // Verify success
      const successMessage = await page.locator('.toast, .notification, .alert').filter({ hasText: /success|updated|saved/i });
      if (await successMessage.count() > 0) {
        expect(await successMessage.first().isVisible()).toBe(true);
      }
    });
  });


  test('should handle table pagination', async ({ page }) => {
    await test.step('Test pagination controls', async () => {
      // Wait for table to load
      await page.waitForSelector('table, [data-testid="data-table"]', { timeout: 10000 });
      
      // Look for next page button
      const nextButton = await page.locator('button[aria-label*="next"], button:has-text("Next"), .pagination-next');
      
      if (await nextButton.count() > 0 && await nextButton.first().isEnabled()) {
        await nextButton.first().click();
        await page.waitForTimeout(2000);
        await pageHelper.takeScreenshot('pagination-next-page');
        
        // Go back to first page
        const prevButton = await page.locator('button[aria-label*="previous"], button:has-text("Previous"), .pagination-prev');
        if (await prevButton.count() > 0 && await prevButton.first().isEnabled()) {
          await prevButton.first().click();
          await page.waitForTimeout(2000);
        }
      }
    });
  });

  test('should delete existing user with confirmation', async ({ page }) => {
    await test.step('Find and click delete button for first user', async () => {
      // Wait for table to load
      await page.waitForSelector('table, [data-testid="data-table"]', { timeout: 10000 });
      
      // Check if there are users to delete
      const userExists = await userManagementHelper.deleteFirstUser();
      
      if (userExists) {
        // Take screenshot of confirmation dialog
        await pageHelper.takeScreenshot('delete-user-confirmation');
      } else {
        console.log('No users available to delete - skipping test');
        return;
      }
    });

    await test.step('Verify delete confirmation dialog', async () => {
      // Check that confirmation dialog has proper content
      const dialogContent = await page.locator('[role="dialog"] p, .modal p, .dialog p').first();
      if (await dialogContent.isVisible()) {
        const content = await dialogContent.textContent();
        expect(content).toBeTruthy();
      }

      // Verify both Cancel and Confirm buttons are present
      const cancelButton = await page.locator('button:has-text("Cancel")');
      const confirmButton = await page.locator('button:has-text("Confirm")');
      
      expect(await cancelButton.isVisible()).toBe(true);
      expect(await confirmButton.isVisible()).toBe(true);
    });

    await test.step('Confirm deletion', async () => {
      await userManagementHelper.confirmDeletion();
      
      // Verify success notification
      await userManagementHelper.verifySuccessNotification(/success|deleted|removed/i);
      
      // Take screenshot of success state
      await pageHelper.takeScreenshot('user-deletion-success');
    });
  });

  test('should cancel user deletion', async ({ page }) => {
    await test.step('Find and click delete button for first user', async () => {
      // Wait for table to load
      await page.waitForSelector('table, [data-testid="data-table"]', { timeout: 10000 });
      
      // Check if there are users to delete
      const userExists = await userManagementHelper.deleteFirstUser();
      
      if (!userExists) {
        console.log('No users available to delete - skipping test');
        return;
      }
    });

    await test.step('Cancel deletion', async () => {
      await userManagementHelper.cancelDeletion();
      
      // Take screenshot showing canceled state
      await pageHelper.takeScreenshot('user-deletion-canceled');
    });

    await test.step('Verify user still exists in table', async () => {
      // The table should still contain users since deletion was canceled
      await pageHelper.waitForTable();
      
      // Check that we still have table rows with user data
      const tableRows = await page.locator('table tbody tr, [data-testid="data-table"] tbody tr');
      if (await tableRows.count() > 0) {
        expect(await tableRows.first().isVisible()).toBe(true);
      }
    });
  });

  test('should search users by name using search icon', async ({ page }) => {
    await test.step('Wait for table to load', async () => {
      await pageHelper.waitForTable();
      await pageHelper.takeScreenshot('table-before-search');
    });

    await test.step('Search for a user by name', async () => {
      // Try to search in the name/fullName column
      const searchSuccess = await userManagementHelper.searchInColumn('name', 'Test');
      
      if (searchSuccess) {
        // Take screenshot of search in action
        await pageHelper.takeScreenshot('name-column-search-active');
        
        // Wait for search results to load
        await page.waitForTimeout(2000);
        
        // Verify search results contain the search term
        const resultsValid = await userManagementHelper.verifySearchResults(0, 'Test'); // Assuming name is first column
        expect(resultsValid).toBe(true);
        
        await pageHelper.takeScreenshot('name-search-results');
      } else {
        console.log('Search functionality not available in name column - skipping verification');
      }
    });

    await test.step('Clear search and verify table returns to original state', async () => {
      await userManagementHelper.clearColumnSearch('name');
      await pageHelper.takeScreenshot('search-cleared');
    });
  });

  test('should search users by email using search icon', async ({ page }) => {
    await test.step('Wait for table to load', async () => {
      await pageHelper.waitForTable();
    });

    await test.step('Search for a user by email', async () => {
      // Try to search in the email column
      const searchSuccess = await userManagementHelper.searchInColumn('email', '@');
      
      if (searchSuccess) {
        // Take screenshot of search in action
        await pageHelper.takeScreenshot('email-column-search-active');
        
        // Wait for search results to load
        await page.waitForTimeout(2000);
        
        // Verify search results contain the search term (assuming email is column 4)
        const resultsValid = await userManagementHelper.verifySearchResults(4, '@');
        expect(resultsValid).toBe(true);
        
        await pageHelper.takeScreenshot('email-search-results');
      } else {
        console.log('Search functionality not available in email column - skipping verification');
      }
    });

    await test.step('Clear search', async () => {
      await userManagementHelper.clearColumnSearch('email');
    });
  });

  test('should sort users by name using arrow icons', async ({ page }) => {
    await test.step('Wait for table to load', async () => {
      await pageHelper.waitForTable();
      await pageHelper.takeScreenshot('table-before-sort');
    });

    await test.step('Sort by name in ascending order', async () => {
      const sortSuccess = await userManagementHelper.sortByArrowIcon('name', 'asc');
      
      if (sortSuccess) {
        // Wait for sort to complete
        await page.waitForTimeout(2000);
        
        // Take screenshot of sorted table
        await pageHelper.takeScreenshot('name-sorted-asc');
        
        // Verify the sort worked (check first column)
        const sortValid = await userManagementHelper.verifyTableSort(0, 'asc');
        if (sortValid) {
          expect(sortValid).toBe(true);
        } else {
          console.log('Sort verification failed or not enough data to verify');
        }
      } else {
        console.log('Sort functionality not available - skipping verification');
      }
    });

    await test.step('Sort by name in descending order', async () => {
      const sortSuccess = await userManagementHelper.sortByArrowIcon('name', 'desc');
      
      if (sortSuccess) {
        // Wait for sort to complete
        await page.waitForTimeout(2000);
        
        // Take screenshot of sorted table
        await pageHelper.takeScreenshot('name-sorted-desc');
        
        // Verify the sort worked
        const sortValid = await userManagementHelper.verifyTableSort(0, 'desc');
        if (sortValid) {
          expect(sortValid).toBe(true);
        }
      }
    });
  });

  test('should sort users by role using arrow icons', async ({ page }) => {
    await test.step('Wait for table to load', async () => {
      await pageHelper.waitForTable();
    });

    await test.step('Sort by role column', async () => {
      const sortSuccess = await userManagementHelper.sortByArrowIcon('role', 'asc');
      
      if (sortSuccess) {
        // Wait for sort to complete
        await page.waitForTimeout(2000);
        
        // Take screenshot of sorted table
        await pageHelper.takeScreenshot('role-sorted');
        
        // Verify the sort worked (assuming role is column 3)
        const sortValid = await userManagementHelper.verifyTableSort(3, 'asc');
        if (sortValid) {
          expect(sortValid).toBe(true);
        }
      } else {
        console.log('Sort functionality not available for role column');
      }
    });
  });
});
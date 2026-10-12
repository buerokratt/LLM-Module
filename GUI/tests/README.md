## Contains both unit and integration test for GUI

### Playwright E2E Tests

This directory contains end-to-end tests using Playwright for the RAG Module GUI.

#### Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Install Playwright browsers:
   ```bash
   npx playwright install
   ```

#### Running Tests

- Run all tests: `npm test`
- Run tests with UI: `npm run test:ui`
- Run tests in headed mode: `npm run test:headed`
- Debug tests: `npm run test:debug`

#### Test Files

- `auth.spec.ts` - Authentication flow tests (requires auth service running)
- `basic-auth.spec.ts` - Basic authentication tests (can run standalone)

#### Authentication Service

The tests expect an authentication service running at `http://localhost:3004/et/dev-auth` (from .env.development).
If the service is not running, some tests will gracefully handle the failure and run alternative scenarios.
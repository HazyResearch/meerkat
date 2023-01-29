// src/mocks/setup.ts
import { API_URL } from '$lib/constants';
import { server } from './server';
import { afterAll, afterEach, beforeAll } from 'vitest'
import { fetch } from 'cross-fetch';

// Add `fetch` polyfill.
global.fetch = fetch;


beforeAll(async () => {
    server.listen({ onUnhandledRequest: 'error' });
    // Set the API_URL to the test server
    API_URL.set('http://test.app');
});
afterAll(() => server.close());
afterEach(() => server.resetHandlers());
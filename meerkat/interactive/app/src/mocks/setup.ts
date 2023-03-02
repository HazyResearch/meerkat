import { API_URL } from '$lib/constants';
import '@testing-library/jest-dom';
import { fetch } from 'cross-fetch';
import { afterAll, afterEach, beforeAll } from 'vitest';
import { server } from './server';


// Add `fetch` polyfill.
global.fetch = fetch;


beforeAll(async () => {
    server.listen({ onUnhandledRequest: 'error' });
    // Set the API_URL to the test server
    API_URL.set('http://test.app');
});
afterAll(() => server.close());
afterEach(() => server.resetHandlers());
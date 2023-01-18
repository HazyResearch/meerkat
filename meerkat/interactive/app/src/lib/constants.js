// Use VITE_API_URL for dev, VITE_API_URL_PLACEHOLDER for prod

import { writable } from "svelte/store";

// TODO: select based on NODE_ENV instead of OR below
export const API_URL = writable(import.meta.env['VITE_API_URL'] || import.meta.env['VITE_API_URL_PLACEHOLDER']);

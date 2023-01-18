import { writable } from "svelte/store";

export const API_URL = writable(import.meta.env['VITE_API_URL'] || import.meta.env['VITE_API_URL_PLACEHOLDER']);
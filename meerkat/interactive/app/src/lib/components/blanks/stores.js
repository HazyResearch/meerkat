import { writable } from "svelte/store";
export const global_stores = new Map();
export const store_lock = writable(new Set());

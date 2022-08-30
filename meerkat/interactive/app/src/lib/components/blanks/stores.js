import { writable } from "svelte/store";
export const global_stores = new Map();
export const backend_updated_stores = writable(new Set());
export const excess_stores = writable(new Set());
export const temp = writable(new Map());
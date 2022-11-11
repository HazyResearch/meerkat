import { writable } from "svelte/store";


// Global application-wide mapping from ref ids to
//  ref ids to trigger reactivity
export let _refs = new Map();

export const _data = writable(new Map());
export const _backend = writable(new Map());
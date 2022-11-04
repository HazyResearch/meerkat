import { writable } from "svelte/store";


// Global application-wide mapping from box ids to
//  box ids to trigger reactivity
export let _boxes = new Map();

export const _data = writable(new Map());
export const _backend = writable(new Map());
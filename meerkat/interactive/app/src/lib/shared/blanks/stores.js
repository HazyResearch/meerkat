import { writable } from "svelte/store";

export function meerkat_writable(value) {
    const { subscribe, set, update } = writable(value);
    const trigger_store = writable(0);

    return {
        subscribe,
        update: update,
        set: (value, trigger = true) => {
            set(value);
            if (trigger) {
                trigger_store.update(n => n + 1);
            };
        },
        trigger_store: trigger_store
    };
}

export const global_stores = new Map();
export const store_lock = writable(new Set());

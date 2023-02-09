import { writable } from "svelte/store";
import { nestedMap } from "./tools";

export function meerkatWritable(value) {
    const { subscribe, set, update } = writable(value);
    const triggerStore = writable(0);
    const storeId = null;
    const backendStore = null;

    return {
        subscribe,
        update: update,
        set: (value, trigger = true) => {
            set(value);
            if (trigger) {
                triggerStore.update(n => n + 1);
            };
        },
        triggerStore: triggerStore,
        storeId: storeId,
        backendStore: backendStore,
    };
};

export const globalStores = new Map();

export const createStoresFromComponent = (component) => {
    return nestedMap(
        component,
        (v) => {
            if (!v) {
                return v;
            }
            if (v.store_id !== undefined) {
                // unpack the store
                if (!globalStores.has(v.store_id)) {
                    // add it to the global_stores Map if it isn't already there
                    let store = meerkatWritable(v.value);
                    store.storeId = v.store_id;
                    // Only stores that have children i.e. are part of the
                    // computation graph are considered to be backend stores
                    // If the store is not a backend store, then its value
                    // will not be synchronized with the backend
                    // Frontend only stores are useful to synchronize values
                    // between frontend components
                    store.backendStore = v.has_children;
                    globalStores.set(v.store_id, store);
                }
                return globalStores.get(v.store_id);
            } else if (v.refId !== undefined) {
                if (!globalStores.has(v.refId)) {
                    // add it to the global_stores Map if it isn't already there
                    globalStores.set(v.refId, writable(v));
                }
                return globalStores.get(v.refId);
            }
            return v;
        }
    );
};

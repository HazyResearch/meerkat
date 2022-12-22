import { nestedMap } from "../../utils/tools";
import { writable } from "svelte/store";

export function meerkat_writable(value) {
    const { subscribe, set, update } = writable(value);
    const trigger_store = writable(0);
    const store_id = null;
    const backend_store = null;

    return {
        subscribe,
        update: update,
        set: (value, trigger = true) => {
            set(value);
            if (trigger) {
                trigger_store.update(n => n + 1);
            };
        },
        trigger_store: trigger_store,
        store_id: store_id,
        backend_store: backend_store,
    };
};

export const global_stores = new Map();

export const create_stores_from_component = (component) => {
    return nestedMap(
        component,
        (v) => {
            if (!v) {
                return v;
            }
            if (v.store_id !== undefined) {
                // unpack the store
                if (!global_stores.has(v.store_id)) {
                    // add it to the global_stores Map if it isn't already there
                    let store = meerkat_writable(v.value);
                    store.store_id = v.store_id;
                    // Only stores that have children i.e. are part of the
                    // computation graph are considered to be backend stores
                    // If the store is not a backend store, then its value
                    // will not be synchronized with the backend
                    // Frontend only stores are useful to synchronize values
                    // between frontend components
                    store.backend_store = v.has_children;
                    global_stores.set(v.store_id, store);
                }
                return global_stores.get(v.store_id);
            } else if (v.ref_id !== undefined) {
                if (!global_stores.has(v.ref_id)) {
                    // add it to the global_stores Map if it isn't already there
                    global_stores.set(v.ref_id, writable(v));
                }
                return global_stores.get(v.ref_id);
            }
            return v;
        }
    );
};

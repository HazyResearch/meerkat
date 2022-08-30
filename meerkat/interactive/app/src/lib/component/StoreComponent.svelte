<script lang="ts">
    import { global_stores, backend_updated_stores } from '$lib/components/blanks/stores';
    import { getContext } from "svelte";

    const { store_trigger } = getContext('Interface')

    export let store_id: string;
    export let store: any;
    export let callback = () => {
        // Check if the set of backend_stores is non-empty 
        if ($backend_updated_stores.size > 0) {
            // The backend just returned a set of stores to be updated
            // as StoreModifications
            
            // Try to pop the store_id off the $backend_updated_stores
            if ($backend_updated_stores.has(store_id)) {
                // store_id exists, pop it off
                $backend_updated_stores.delete(store_id);
                // timestamp is incorrect here for the backend driven store update
                // $temp.set(store_id, {value: $store, timestamp: Date.now()})
            } else {
                console.log("Y2K");
                // // already popped off, this store needs to be checked for a possible backend retrigger
                // $excess_stores.add(store_id);
                // // This is incorrect, fix
                // $temp.set(store_id, {value: $store, timestamp: Date.now()})
            }
        } else {
            // TODO: Check if this is a backend facing store!!!!
            // Do nothing, call the store trigger function to execute the 
            // computational graph on the relevant nodes on the backend
            store_trigger(store_id, $store);
        }
    };

    $: $store, callback();
</script>
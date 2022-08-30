<script lang="ts">
    import { store_lock } from '$lib/components/blanks/stores';
    import { getContext } from "svelte";

    const { store_trigger } = getContext('Interface');

    export let store_id: string;
    export let store: any;

    // Callback that runs when the store changes
    export let callback = () => {
        // Check if the set of backend_stores is non-empty 
        if ($store_lock.size > 0) {
            // The backend just returned a set of stores to be updated
            // as StoreModifications
            
            // Try to pop the store_id off the $backend_updated_stores
            if ($store_lock.has(store_id)) {
                // store_id exists, pop it off
                $store_lock.delete(store_id);
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
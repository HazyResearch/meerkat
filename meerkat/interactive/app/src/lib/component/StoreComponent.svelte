<script lang="ts">
	import { store_lock } from '$lib/components/blanks/stores';
	import { getContext } from 'svelte';

	const { store_trigger } = getContext('Interface');

	export let store_id: string;
	export let store: any;
	export let is_backend_store: boolean;

    // this assumes that all the stores are created with meerkat_writable 
    let trigger_store = store.trigger_store;

	// Callback that runs when the store changes
	export let callback = () => {
       
		if (!is_backend_store) {
			return;
		}
        $store_trigger(store_id, $store);
	};

    // only respond to trigger_store
	$: $trigger_store, callback();
</script>

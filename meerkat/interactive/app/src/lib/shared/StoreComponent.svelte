<script lang="ts">
	import { store_trigger } from '$lib';

	export let store_id: string;
	export let store: any;
	export let is_backend_store: boolean;

	// KG: temporarily make all Store objects backend stores
	// The check for undefined allows us to ensure that we don't
	// set this for other NodeMixin objects (i.e. non-Store objects like DataFrames)
	if (is_backend_store !== undefined) {
		is_backend_store = true;
	}

	// this assumes that all the stores are created with meerkat_writable
	let trigger_store = store.trigger_store;

	// Callback that runs when the store changes
	export let callback = () => {
		if (!is_backend_store) {
			return;
		}
		store_trigger(store_id, $store);
	};

	// only respond to trigger_store
	$: $trigger_store, callback();
</script>

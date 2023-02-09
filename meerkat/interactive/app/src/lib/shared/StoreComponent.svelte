<script lang="ts">
	import { updateStore } from '$lib/utils/api';

	export let storeId: string;
	export let store: any;
	export let isBackendStore: boolean;

	// KG: We make all Store objects backend stores
	// This means that Stores will always be synced with the backend
	// The check for undefined ensures we don't
	// set this for other NodeMixin objects (i.e. non-Store objects like DataFrames)
	if (isBackendStore !== undefined) {
		isBackendStore = true;
	}

	// this assumes that all the stores are created with meerkat_writable
	let triggerStore = store.triggerStore;

	let _mounted = false;
	// Callback that runs when the store changes
	export let callback = () => {
		if (!_mounted) {
			_mounted = true;
			return;
		}
		if (!isBackendStore) {
			return;
		}
		updateStore(storeId, $store);
	};

	// only respond to triggerStore
	$: $triggerStore, callback();
</script>

<script lang="ts">
	import type { PageType } from '$lib/utils/types';
	import { Toaster } from 'svelte-french-toast';
	import { createStoresFromComponent, globalStores } from '../utils/stores';
	import DynamicComponent from './DynamicComponent.svelte';
	import StoreComponent from './StoreComponent.svelte';

	export let config: PageType;

	$: {
		if (config.component) {
			config.component = createStoresFromComponent(config.component);
		}
	}
</script>

{#each Array.from(globalStores.keys()) as storeId}
	<StoreComponent
		{storeId}
		store={globalStores.get(storeId)}
		isBackendStore={globalStores.get(storeId).backendStore}
	/>
{/each}

{#if config.component}
	<DynamicComponent {...config.component} />
{:else}
	<slot />
{/if}

<Toaster />

<script lang="ts">
	import type { InterfaceType } from '$lib/utils/types';
	import { create_stores_from_component, global_stores } from './blanks/stores';
	import DynamicComponent from './DynamicComponent.svelte';
	import StoreComponent from './StoreComponent.svelte';

	export let config: InterfaceType;
    
	$: {
		if (config.component) {
			config.component = create_stores_from_component(config.component);
		};
	}
	
	// Setup context
	
</script>

{#each Array.from(global_stores.keys()) as store_id}
    <StoreComponent
        {store_id}
        store={global_stores.get(store_id)}
        is_backend_store={global_stores.get(store_id).backend_store}
    />
{/each}

{#if config.component}
	<DynamicComponent {...config.component} />
{:else}
	<slot />
{/if}

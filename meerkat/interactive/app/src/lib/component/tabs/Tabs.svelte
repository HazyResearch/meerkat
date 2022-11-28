<script lang="ts">
    import DynamicComponent from '$lib/shared/DynamicComponent.svelte';
    import type {Component} from '$lib/utils/types';

	interface Tab {
		label: string;
		id: string;
        component: Component
	}

	export let tabs: Array<Tab> = [];
    console.log(tabs);
    
    let activeTabId = '';

	function setActiveTab(id: string): void {
		if (id === activeTabId) {
			activeTabId = '';
		} else {
			activeTabId = id;
		}
	}

</script>

<div class="flex">
	<div class="flex-grow flex justify-center">
		{#each tabs as { label, id }}
			<button
				class="{activeTabId === id
					? 'active inline-block py-1 px-4 text-lg font-medium text-center text-white bg-violet-600 rounded-lg'
					: 'inline-block py-1 px-4 text-lg font-medium text-center text-gray-500 rounded-lg hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-white'} "
				on:click={() => setActiveTab(id)}
			>
				{label}
			</button>
		{/each}
	</div>
</div>

{#each tabs as {id, component}}
    {#if id === activeTabId}
        <DynamicComponent name={component.name} props={component.props} />
    {/if}
{/each}
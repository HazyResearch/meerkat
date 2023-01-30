<script lang="ts">
	import { setContext } from 'svelte';
	import { writable } from 'svelte/store';
    import Toggle from '$lib/shared/deprecate/common/Toggle.svelte';
    import banner from "$lib/assets/banner_small.png"
	import { activeTabId } from './stores';

    export let toggle_button: boolean = false; 

	interface TabInterface {
		label: string;
		id: string;
	}

	let tabs: Array<TabInterface> = [];

	const tabsContext = setContext('tabs', { activeTabId, addTab });

	function addTab(label: string, id: string): void {
		tabs = [...tabs, { label, id }];
	}
	function setActiveTab(id: string): void {
		if (id === $activeTabId) {
			activeTabId.set('');
		} else {
			activeTabId.set(id);
		}
	}

</script>

<div class="flex">
    <div class="flex-none w-50 pl-7">
        <img src={banner} alt="Meerkat" class="h-10"/>
    </div>
	<div class="flex-grow flex justify-center">
		{#each tabs as { label, id }}
			<button
				class="{$activeTabId === id
					? 'active inline-block py-1 px-4 text-lg font-medium text-center text-white bg-violet-600 rounded-lg'
					: 'inline-block py-1 px-4 text-lg font-medium text-center text-gray-500 rounded-lg hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-white'} "
				on:click={() => setActiveTab(id)}
			>
				{label}
			</button>
		{/each}
	</div>
    <div class="flex-none w-50">
        <Toggle label_left="Table" label_right="Gallery" bind:checked={toggle_button} />

    </div>
</div>

<div class="py-3">
    <slot>placeholder</slot>
</div>
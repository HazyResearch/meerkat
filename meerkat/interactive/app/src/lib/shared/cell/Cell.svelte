<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';
	import { writable } from 'svelte/store';

	export let data: any;
	export let cell_component: string = '';
	export let cell_props: object = {};
	export let cell_data_prop: string = 'data';
	export let editable: boolean = false;
	export let column: string = '';

	// need to actually create a new object, since we don't want to modify the
	// cell_props that were passed in

	$: {
		cell_props = {...cell_props};
		cell_props[cell_data_prop] = data;

		// iterate over cell_props and turn them into stores if they aren't already
		for (const [key, value] of Object.entries(cell_props)) {
			if (value === null || value.subscribe === undefined) {
				cell_props[key] = writable(value);
			}
		}
	}

	const dispatch = createEventDispatcher();

	function edit() {
		dispatch('edit', {
			value: data
		});
	}


</script>

<DynamicComponent name={cell_component} props={cell_props} />

<!-- {#if cell_component === 'image'}
	<Image {data} {...cell_props} />
{:else if cell_component === 'code'}
	<Code {data} {...cell_props} />
{:else if cell_component === 'website'} 
	<Website {data} {...cell_props} />
{:else}
	{#if editable}
		<input 				
			class="input input-bordered grow h-7 px-3 rounded-md shadow-md"
			on:change={edit} 
			bind:value={data} 
		/>
	{:else}
		<BasicType {data} {...cell_props} />
	{/if}
{/if} -->

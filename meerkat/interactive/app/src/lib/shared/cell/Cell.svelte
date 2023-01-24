<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';
	// import Code from './code/Code.svelte';
	// import Image from './image/Image.svelte';
	// import BasicType from './basic/Basic.svelte';
	// import Website from './website/website.svelte';

	export let data: any;
	export let column: string = null;
	export let cell_component: string = '';
	export let cell_props: object = {};
	export let editable: boolean = false;

	cell_props.url = data;
	console.log(cell_component);

	const dispatch = createEventDispatcher();

	function edit() {
		dispatch('edit', {
			value: data
		});
	}
</script>

hello
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

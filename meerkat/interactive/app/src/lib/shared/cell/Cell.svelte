<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	
	import Code from './code/Code.svelte';
	import Image from './image/Image.svelte';
	import BasicType from './basic/Basic.svelte';

	export let data: any;
	export let column: string = null;
	export let cell_component: string = '';
	export let cell_props: object = {};
	export let editable: boolean = false;

	const dispatch = createEventDispatcher();

	function edit() {
		dispatch('edit', {
			value: data
		});
	}
</script>

{#if cell_component === 'image'}
	<Image {data} {...cell_props} />
{:else if cell_component === 'code'}
	<Code {data} {...cell_props} />
{:else}
	{#if editable}
		<!-- TODO(Sabri): Make this work with the BasicType formatting -->
		<input 				
			class="input input-bordered grow h-7 px-3 rounded-md shadow-md"
			on:change={edit} 
			bind:value={data} 
		/>
	{:else}
		<BasicType {data} {...cell_props} />
	{/if}
{/if}

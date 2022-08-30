<script context="module" lang="ts">
	import { createEventDispatcher } from 'svelte';

	export interface CellInterface {
		data: any;
		cell_component?: string;
		cell_props?: object;
	}

</script>

<script lang="ts">
	import Code from './code/Code.svelte';
	import Image from './image/Image.svelte';

	export let data: any;
	export let cell_component: string = '';
	export let cell_props: object = {};

	const dispatch = createEventDispatcher();

	function edit() {
		console.log(`editing: ${data}`)
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
	<input on:change={edit} bind:value={data} />
{/if}

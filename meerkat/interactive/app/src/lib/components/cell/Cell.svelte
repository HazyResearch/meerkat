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
	import BasicType from './basic/Basic.svelte';

	export let data: any;
	export let cell_component: string = '';
	export let cell_props: object = {};

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
	<BasicType {data} {...cell_props}/>
	<!-- <input class="bg-transparent w-fit" on:change={edit} bind:value={data} /> -->
{/if}

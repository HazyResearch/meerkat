<script lang="ts">
	import Cell,{ type CellInterface } from '$lib/components/cell/Cell.svelte';
	import { zip } from 'underscore';
	import Pill from '$lib/components/common/Pill.svelte';

	export let pivot: CellInterface;
	export let pivot_header: string;
	export let content: Array<any>;
	export let content_headers: Array<string> = [];

	// Give the card the `flex-grow` Tailwind class to horizontally
	// fill out space in the (containing) flex container.
	export let card_flex_grow: boolean = false;
	export let as_modal: boolean = false;
	export let wrap_content: boolean = false;
</script>

<div class="bg-gray-100 p-2 rounded-lg grid-rows-2 overflow-x-hidden" class:flex-grow={card_flex_grow} class:modal={as_modal}>
	<div class="self-center">
		<Pill header={pivot_header} layout={'wide-header'} />
		<Cell {...pivot} />
	</div>
	<div class="overflow-y-scroll mt-8 p-2 border-t-2 border-solid" class:wrap-content={wrap_content}>
		{#each zip(content_headers, content) as [header, subcontent], j}
			<Pill {header} content={subcontent} layout={'wide-content'} />
		{/each}
	</div>
</div>

<style>
	.modal {
		@apply max-h-[95vh] max-w-[90vw] z-20;
	}

	.wrap-content {
		@apply flex flex-wrap;
	}
</style>

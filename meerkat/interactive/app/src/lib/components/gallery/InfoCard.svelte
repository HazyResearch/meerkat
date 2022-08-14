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

<div class="card" class:flex-grow={card_flex_grow} class:modal={as_modal}>
	<div class="pivot">
		<Pill header={pivot_header} layout={'wide-header'} />
		<Cell {...pivot} />
	</div>
	<div class="content" class:wrap-content={wrap_content}>
		{#each zip(content_headers, content) as [header, subcontent], j}
			<Pill {header} content={subcontent} layout={'wide-content'} />
		{/each}
	</div>
</div>

<style>
	.modal {
		@apply max-h-[95vh] max-w-[90vw];
	}

	.card {
		/* min-width: var(--card-width, ''); */
		@apply bg-gray-400;
		@apply p-2 rounded-lg;
		@apply flex flex-col;
		@apply overflow-y-auto overflow-x-hidden;
	}

	.pivot {
		@apply self-center;
	}

	.content {
		/* Take up atleast 30% of the height of the card */
		@apply min-h-[30%];
		@apply mt-8 p-2 border-t-2 border-solid;
		@apply overflow-y-auto;
	}

	.wrap-content {
		@apply flex flex-wrap;
	}
</style>

<script lang="ts">
	import Cell, { type CellInterface } from '$lib/components/cell/Cell.svelte';
import { zip } from 'underscore';

	export let pivot: CellInterface;
	export let pivot_header: string;
	export let content: any;
	export let content_headers: Array<string> = [];

	// Give the card the `flex-grow` Tailwind class to horizontally
	// fill out space in the (containing) flex container.
	export let card_flex_grow: boolean = false;
</script>

<div class="card" class:flex-grow={card_flex_grow}>
	<div class="pivot">
		<div class="header"><Cell data={pivot_header} /></div>
		<Cell {...pivot} />
	</div>
	<div class="content">
		{#each zip(content_headers, content) as [header, subcontent], j}
			<div class="tag">
				<div class="header"><Cell data={header} /></div>
				<div class="subcontent"><Cell data={subcontent} /></div>
			</div>
		{/each}
	</div>
</div>

<style>
	.card {
		/* min-width: var(--card-width, ''); */
		@apply bg-gray-400;
		@apply p-2 rounded-lg;
		@apply flex flex-col;
	}

	.pivot {
		@apply self-center;
	}

	.content {
		@apply flex flex-wrap items-start mt-2;
	}

	.tag {
		@apply flex flex-row pr-2 pl-2;
	}

	.header {
		@apply p-1 mr-0 ml-0 m-1 rounded-l-md bg-slate-900 items-start;
		@apply text-center text-xs text-ellipsis whitespace-nowrap select-none font-mono;
	}

	.subcontent {
		@apply p-1 ml-0 m-1 rounded-r-md;
		@apply text-center overflow-hidden text-xs text-ellipsis whitespace-nowrap select-none font-mono;
		@apply text-slate-200 bg-slate-500;
	}
</style>

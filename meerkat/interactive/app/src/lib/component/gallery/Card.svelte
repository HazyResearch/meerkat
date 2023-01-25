<script lang="ts">
	import Cell from '$lib/shared/cell/Cell.svelte';
	import type { CellInterface } from '$lib/utils/types';
	import { createEventDispatcher } from 'svelte';
	import Selected from './Selected.svelte';
	import { getContext } from 'svelte';

	export let keyidx: string;
	export let posidx: number;
	export let pivot: CellInterface;
	export let content: Array<CellInterface>;
	export let layout: string;
	export let selected: boolean = false;

	console.log(pivot)

	const open_row_modal: Function = getContext('open_row_modal');
	const dispatch = createEventDispatcher();
</script>

<div
	class="mx-2 my-4"
	class:card-masonry={layout === 'masonry'}
	class:card-gimages={layout === 'gimages'}
>
	<!-- Pivot (main) element -->
	<div class="relative">
		{#if selected}
			<div class="absolute top-2 left-2 z-10">
				<Selected />
			</div>
		{/if}
		<!-- Pivot item -->
		<div
			class="pivot"
			class:selected-pivot={selected}
			on:dblclick={(e) => {
				open_row_modal(posidx);
			}}
			on:click={(e) => {
				dispatch('click', e);
			}}
		>
			<Cell {...pivot} />
		</div>
	</div>

	<!-- Content -->
	<div class="content">
		{#each content as subcontent, j}
			<div
				class="subcontent mx-1 my-1 px-2 py-0.5 rounded-md text-left text-slate-800 text-xs bg-slate-200"
			>
				<div class="font-bold font-mono whitespace-nowrap overflow-hidden text-ellipsis">
					{subcontent.column}
				</div>
				<div class="font-mono whitespace-nowrap overflow-hidden text-ellipsis">
					<Cell {...subcontent} />
				</div>
			</div>
		{/each}
	</div>
</div>

<style>
	.card {
		min-width: var(--card-width, '');
		@apply m-2 border-2 border-solid rounded-lg shadow-md;
		@apply dark:border-gray-600;
	}

	.card-masonry {
		/* Solution 1: multiple columns in a masonry layout */
		@apply break-inside-avoid h-auto;
	}

	.card-gimages {
		/* Solution 2: flex containers in the Google Images style */
		/* Make the card a flex-col so the pivot element can be centered horizontally */
		@apply flex flex-col;
	}

	.pivot {
		@apply self-center;
	}

	.content {
		/* Row format for tags */
		@apply flex flex-wrap items-start;
	}

	.selected-pivot {
		@apply opacity-50;
	}

	.pivot:hover {
		@apply opacity-50;
	}
</style>

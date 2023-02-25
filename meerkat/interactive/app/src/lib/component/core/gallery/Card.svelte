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
	export let height: number = 12;
	export let selected: boolean = false;

	const open_row_modal: Function = getContext('open_row_modal');
	const dispatch = createEventDispatcher();

</script>

<div
	class="mx-2 my-4"
	class:card-masonry={layout === 'masonry'}
	class:card-gimages={layout === 'gimages'}
>
	<!-- Pivot (main) element -->
	<div class="relative  overflow-hidden">
		{#if selected}
			<div class="absolute top-2 left-2 z-10">
				<Selected />
			</div>
		{/if}
		<!-- Pivot item -->
		<button
			class="self-center rounded-sm hover:opacity-75 bg-white shadow-sm  overflow-hidden"
			style:height={`${height}vh`}
			class:opacity-50={selected}
			on:dblclick={(e) => {
				open_row_modal(posidx);
			}}
			on:click={(e) => {
				dispatch('click', e);
			}}
		>
			<Cell {...pivot} />
		</button>
	</div>

	<!-- Content -->
	{#if height >= 15}
		<div class="flex flex-wrap items-start w-full">
			{#each content as subcontent}
				<div
					class="subcontent mx-1 my-1 px-2 py-0.5 rounded-md text-left text-slate-800 text-xs bg-slate-200"
				>
					<div class="font-bold font-mono whitespace-nowrap overflow-hidden text-ellipsis">
						{subcontent.column}
					</div>
					<div class="font-mono whitespace-nowrap overflow-hidden text-ellipsis rounded-md">
						<Cell {...subcontent} />
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.card-masonry {
		/* Solution 1: multiple columns in a masonry layout */
		@apply break-inside-avoid h-auto;
	}

	.card-gimages {
		/* Solution 2: flex containers in the Google Images style */
		/* Make the card a flex-col so the pivot element can be centered horizontally */
		@apply flex flex-col;
	}
</style>

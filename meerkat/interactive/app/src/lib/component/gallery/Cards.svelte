<script lang="ts">
	import type { DataFrameChunk, ColumnInfo, DataFrameSchema } from '$lib/api/dataframe';
	import { without } from 'underscore';
	import Card from './Card.svelte';

	export let schema: DataFrameSchema;
	let column_infos: Array<ColumnInfo> = schema.columns;
	export let chunk: DataFrameChunk;

	export let layout = 'gimages'; // 'gimages' or 'masonry'
	export let layout_style = 'natural'; // 'natural' or 'square'

	// Main column to display.
	export let main_column: string = 'image';

	// Columns to use for tags in the GalleryView.
	export let tag_columns: Array<string> = [];

	// Columns size to display
	export let cell_size: number = 6;

	// Selected rows
	export let selected: Array<string> = [];


	$: pivot_height = layout === 'gimages' ? cell_size : undefined;
	$: num_columns = layout === 'masonry' ? cell_size : undefined;
	
</script>

<div
	class="dark:bg-slate-600 h-fit"
	class:panel-masonry={layout === 'masonry'}
	class:panel-gimages={layout === 'gimages'}
	style:columns={layout === 'gimages' ? null : num_columns}
>
	{#each chunk.keyidxs as keyidx, i}
		<Card
			{keyidx}
			posidx={chunk.posidxs[i]}
			pivot={chunk.get_cell(i, main_column)}
			content={tag_columns.map((column) => chunk.get_cell(i, column))}
			{layout}
			bind:height={pivot_height}
			selected={selected.includes(keyidx)}
			on:click={(e) => {
				if (e.detail.shiftKey) {
					if (selected.length === 0) {
						selected.push(keyidx);
					} else {
						let last_idx = selected[selected.length - 1];
						let last_i = chunk.keyidxs.indexOf(last_idx);
						let i = chunk.keyidxs.indexOf(keyidx);
						if (i > last_i) {
							for (let j = last_i; j <= i; j++) {
								if (!selected.includes(chunk.keyidxs[j])) {
									selected.push(chunk.keyidxs[j]);
								}
							}
						} else {
							for (let j = last_i; j >= i; j--) {
								if (!selected.includes(chunk.keyidxs[j])) {
									selected.push(chunk.keyidxs[j]);
								}
							}
						}
					}
				} else if (e.detail.altKey) {
					selected = [];
					selected.push(keyidx);
				} else {
					if (selected.includes(keyidx)) {
						selected = without(selected, keyidx);
					} else if (!selected.includes(keyidx)) {
						selected.push(keyidx);
					}
				}
				selected = selected;
			}}
		>
		</Card>
	{/each}
</div>

<style>
	.threshold_selector--clicked {
		@apply bg-violet-400;
	}

	.panel-masonry {
		/* Solution 1: multiple columns in a masonry layout */
		@apply columns-10;
	}

	.panel-gimages {
		/* Solution 2: flex containers in the Google Images style */
		@apply flex flex-wrap justify-around;
	}
</style>

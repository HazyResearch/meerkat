<script lang="ts">
	import type { DataFrameChunk } from '$lib/utils/dataframe';
	import { without } from 'underscore';
	import Card from './Card.svelte';

	export let chunk: DataFrameChunk;
	export let layout = 'gimages'; // 'gimages' or 'masonry'

	/** Main column to display. */
	export let mainColumn: string = 'image';

	/** Columns to use for tags in the GalleryView. */
	export let tagColumns: Array<string> = [];

	/** Columns size to display. */
	export let cellSize: number = 6;

	/** Selected rows. */
	export let selected: Array<string> = [];

	/** Whether to allow selection. */
	export let allowSelection: boolean = false;

	$: pivotHeight = layout === 'gimages' ? cellSize : undefined;
	$: numColumns = layout === 'masonry' ? cellSize : undefined;
</script>

<div
	class="h-fit"
	class:panel-masonry={layout === 'masonry'}
	class:panel-gimages={layout === 'gimages'}
	style:columns={layout === 'gimages' ? null : numColumns}
>
	{#each chunk.keyidxs as keyidx, i}
		<Card
			{keyidx}
			posidx={chunk.posidxs[i]}
			pivot={chunk.getCell(i, mainColumn)}
			content={tagColumns.map((column) => chunk.getCell(i, column))}
			{layout}
			bind:height={pivotHeight}
			selected={selected.includes(keyidx)}
			on:click={(e) => {
				if (!allowSelection) {
					return;
				}
				if (e.detail.shiftKey) {
					if (selected.length === 0) {
						selected.push(keyidx);
					} else {
						let lastIdx = selected[selected.length - 1];
						let lasti = chunk.keyidxs.indexOf(lastIdx);
						let i = chunk.keyidxs.indexOf(keyidx);
						if (i > lasti) {
							for (let j = lasti; j <= i; j++) {
								if (!selected.includes(chunk.keyidxs[j])) {
									selected.push(chunk.keyidxs[j]);
								}
							}
						} else {
							for (let j = lasti; j >= i; j--) {
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
		/>
	{/each}
</div>

<style>
	.panel-masonry {
		/* Solution 1: multiple columns in a masonry layout */
		@apply columns-10;
	}

	.panel-gimages {
		/* Solution 2: flex containers in the Google Images style */
		@apply flex flex-wrap justify-around;
	}
</style>

<script lang="ts">
	import type { DataFrameChunk, ColumnInfo, DataFrameSchema } from '$lib/api/dataframe';
	import { writable, type Writable } from 'svelte/store';
	import { get, map, without } from 'underscore';
	import Card from './Card.svelte';
	import InfoModal from './InfoModal.svelte';

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

	// Selected columns
	export let selected: Array<string> = [];

	let columns = column_infos.map((col: any) => col.name);
	let column_components = column_infos.map((col: any) => col.cell_component);

	let tag_indices: Array<number> = tag_columns.map((tag) => columns.indexOf(tag));
	let main_index: number = columns.indexOf(main_column);
	let pivot_cell_component = column_components[main_index];

	// need to get the row index of the primary key
	let primary_key_index: number;
	if (schema.primary_key !== undefined) {
		primary_key_index = columns.findIndex((c) => c === schema.primary_key);
	}

	$: pivot_height = layout === 'gimages' ? cell_size : undefined;
	$: num_columns = layout === 'masonry' ? cell_size : undefined;

	// eventually everything will have to be done based on primary key, but placeholder
	// for now
	const get_keyidx = (row: Array<any>, i: number) => {
		if (schema.primary_key === undefined) {
			return i;
		} else {
			return row[primary_key_index];
		}
	};

	$: keyidxs = chunk.rows.map(get_keyidx);
</script>

<div class="h-full">
	<div
		class="dark:bg-slate-600 h-full"
		class:panel-masonry={layout === 'masonry'}
		class:panel-gimages={layout === 'gimages'}
		style:columns={layout === 'gimages' ? null : num_columns}
	>
		{#each chunk.rows.map((row, i) => [row, keyidxs[i], i]) as [row, keyidx, i]}
			<Card
				{keyidx}
				posidx={chunk.indices[i]}
				pivot={{
					data: row[main_index],
					cell_component: pivot_cell_component,
					cell_props: {
						height: layout === 'gimages' ? pivot_height : '',
						width: layout_style === 'square' ? pivot_height : ''
					}
				}}
				content={pivot_height >= 15 || num_columns <= 6
					? tag_indices.map((z) => ({ data: row[z], column: columns[z] }))
					: []}
				{layout}
				card_flex_grow={false}
				selected={selected.includes(keyidx)}
				on:click={(e) => {
					if (e.detail.shiftKey) {
						if (selected.length === 0) {
							selected.push(keyidx);
						} else {
							let last_idx = selected[selected.length - 1];
							let last_i = keyidxs.indexOf(last_idx);
							let i = keyidxs.indexOf(keyidx);
							if (i > last_i) {
								for (let j = last_i; j <= i; j++) {
									if (!selected.includes(keyidxs[j])) {
										selected.push(keyidxs[j]);
									}
								}
							} else {
								for (let j = last_i; j >= i; j--) {
									if (!selected.includes(keyidxs[j])) {
										selected.push(keyidxs[j]);
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
				<div slot="pivot-tooltip">Double-click to see row</div>
			</Card>
		{/each}
	</div>
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

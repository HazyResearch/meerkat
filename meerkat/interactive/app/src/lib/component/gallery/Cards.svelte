<script lang="ts">
	import type { DataPanelRows, ColumnInfo, DataPanelSchema } from '$lib/api/datapanel';
	import { writable, type Writable } from 'svelte/store';
	import { get, map, without } from 'underscore';
	import Card from './Card.svelte';
	import InfoModal from './InfoModal.svelte';

	export let schema: DataPanelSchema;
	let column_infos: Array<ColumnInfo> = schema.columns;
	export let rows: DataPanelRows | null;
	export let primary_key: string;

	export let layout = 'gimages'; // 'gimages' or 'masonry'
	export let layout_style = 'natural'; // 'natural' or 'square'

	// Main column to display.
	export let main_column: string = 'image';

	// Columns to use for tags in the GalleryView.
	export let tag_columns: Array<string> = [];

	// Columns size to display
	export let cell_size: number = 6;

	// Columns to display
	export let selected: Writable<Array<number>> = writable([]);

	let columns = column_infos.map((col: any) => col.name);
	let column_components = column_infos.map((col: any) => col.cell_component);

	let tag_indices: Array<number> = tag_columns.map((tag) => columns.indexOf(tag));
	let main_index: number = columns.indexOf(main_column);
	let pivot_cell_component = column_components[main_index];
	
	// need to get the row index of the primary key 
	let primary_key_index: number;
	if (primary_key !== undefined) {
		primary_key_index = columns.findIndex((c) => c === primary_key);
	} 

	$: pivot_height = layout === 'gimages' ? cell_size : undefined;
	$: num_columns = layout === 'masonry' ? cell_size : undefined;

	// eventually everything will have to be done based on primary key, but placeholder 
	// for now
	const get_idx = (row: Array<any>, i: number) => {
		if (primary_key === undefined) {
			return i;
		} else {
			return row[primary_key_index];
		}
	};

	let idxs = rows.rows.map(get_idx)
</script>

<div class="h-full">
	<div
		class="dark:bg-slate-600 h-full"
		class:panel-masonry={layout === 'masonry'}
		class:panel-gimages={layout === 'gimages'}
		style:columns={layout === 'gimages' ? null : num_columns}
	>
		{#each rows.rows.map((row, i) => [row, idxs[i]]) as [row, idx]}
			<Card
				id={idx}
				pivot={{
					data: row[main_index],
					cell_component: pivot_cell_component,
					cell_props: {
						height: layout === 'gimages' ? `${pivot_height}vh` : '',
						width: layout_style === 'square' ? `${pivot_height}vh` : ''
					}
				}}
				content={pivot_height >= 15 || num_columns <= 6
					? tag_indices.map((z) => ({ data: row[z] }))
					: []}
				{layout}
				pivot_tooltip={true}
				content_tooltip={true}
				card_flex_grow={false}
				--card-width=""
				pivot_modal_component={InfoModal}
				pivot_modal_component_props={{
					props: {
						pivot: {
							data: row[main_index],
							cell_component: pivot_cell_component,
							cell_props: { height: '100%' }
						},
						pivot_header: main_column,
						content: row.filter((z, i) => i !== main_index),
						content_headers: columns.filter((z, i) => i !== main_index),
						card_flex_grow: false
					}
				}}
				selected={$selected.includes(idx)}
				on:click={(e) => {
					if (e.detail.shiftKey) {
						if ($selected.length === 0) {
							$selected.push(idx);
						} else {
							let last_idx = $selected[$selected.length - 1];
							let last_i = idxs.indexOf(last_idx);
							let i = idxs.indexOf(idx);
							if (i > last_i) {
								for (let j = last_i; j <= i; j++) {
									if (!$selected.includes(idxs[j])) {
										$selected.push(idxs[j]);
									}
								}
							} else {
								for (let j = last_i; j >= i; j--) {
									if (!$selected.includes(idxs[j])) {
										$selected.push(idxs[j]);
									}
								}
							}
						}
					} else if (e.detail.altKey) {
						$selected = [];
						$selected.push(idx);
					} else {
						if ($selected.includes(idx)) {
							$selected = without($selected, idx);
						} else if (!$selected.includes(idx)) {
							$selected.push(idx);
						}
					}
					$selected = $selected;
				}}
			>
				<div slot="pivot-tooltip">Double-click to see example</div>
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

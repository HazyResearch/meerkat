<script lang="ts">
	import type { DataPanelRows, ColumnInfo, DataPanelSchema } from '$lib/api/datapanel';
	import Slider from '$lib/components/common/Slider.svelte';
	import { includes, isNumber, without } from 'underscore';
	import Card from './Card.svelte';
	import InfoModal from './InfoModal.svelte';

	export let schema: DataPanelSchema;
	let column_infos: Array<ColumnInfo> = schema.columns;
	export let rows: DataPanelRows | null;

	export let layout = 'gimages'; // 'gimages' or 'masonry'
	export let layout_style = 'natural'; // 'natural' or 'square'

	// Main column to display.
	export let main_column: string = 'image';

	// Columns to use for tags in the GalleryView.
	export let tag_columns: Array<string> = [];

	// Columns size to display 
	export let cell_size: number = 6;

	let columns = column_infos.map((col: any) => col.name);
	let column_components = column_infos.map((col: any) => col.cell_component);

	let tag_indices: Array<number> = tag_columns.map((tag) => columns.indexOf(tag));
	let main_index: number = columns.indexOf(main_column);
	let pivot_cell_component = column_components[main_index];

	$: pivot_height = layout === 'gimages' ? cell_size : undefined;
	$: num_columns = layout === 'masonry' ? cell_size : undefined;


	// Temporary code for labeling
	let selected_indices: Array<number> = [];
</script>



<div class="h-full">
	<div
		class="dark:bg-slate-600 h-full"
		class:panel-masonry={layout === 'masonry'}
		class:panel-gimages={layout === 'gimages'}
		style:columns={layout === 'gimages' ? null : num_columns}
	>
		{#each rows.rows as row, i}
			<Card
				id={i.toString()}
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
				on:click={() => {
					if (selected_indices.includes(i)) {
						selected_indices = without(selected_indices, i);
					} else if (!selected_indices.includes(i) && selected_indices.length < 2) {
						selected_indices.push(i); 
						selected_indices.sort((a, b) => a - b);
					}
					selected_indices = selected_indices;
				}}			
			>
				<div slot="pivot-tooltip">Double-click to see example</div>
			</Card>
		{/each}
	</div>
	
</div>

<style>
	.threshold_selector--clicked{
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

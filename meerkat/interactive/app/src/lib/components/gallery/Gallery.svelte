<script lang="ts">
	import type { DataPanelRows, ColumnInfo, DataPanelSchema } from '$lib/api/datapanel';
	import Slider from '$lib/components/common/Slider.svelte';
	import { includes, without } from 'underscore';
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

	let columns = column_infos.map((col: any) => col.name);
	let column_components = column_infos.map((col: any) => col.cell_component);

	let tag_indices: Array<number> = tag_columns.map((tag) => columns.indexOf(tag));
	let main_index: number = columns.indexOf(main_column);
	let pivot_cell_component = column_components[main_index];

	let cell_size_variable = layout === 'gimages' ? 20 : 4;
	$: pivot_height = layout === 'gimages' ? cell_size_variable : undefined;
	$: num_columns = layout === 'masonry' ? cell_size_variable : undefined;


	// Temporary code for labeling
	let selected_indices: Array<number> = [];
</script>


<div class="flex mx-2 w-52 items-center">
	<div class="inline-flex flex-shrink-0 justify-center items-center w-8 h-8 rounded-lg">
		<svg
			xmlns="http://www.w3.org/2000/svg"
			width="18"
			height="18"
			class=" text-slate-500 fill-current self-center"
			viewBox="0 0 20 20"
		>
			<path d="M6.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z" />
			<path
				d="M2.002 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2h-12zm12 1a1 1 0 0 1 1 1v6.5l-3.777-1.947a.5.5 0 0 0-.577.093l-3.71 3.71-2.66-1.772a.5.5 0 0 0-.63.062L1.002 12V3a1 1 0 0 1 1-1h12z"
			/>
		</svg>
	</div>

	<Slider
		id="range1"
		min={layout === 'gimages' ? 6 : 1}
		max={layout === 'gimages' ? 48 : 6}
		bind:value={cell_size_variable}
	/>
	<div class="inline-flex flex-shrink-0 justify-center items-center w-8 h-8 rounded-lg">
		<svg
			xmlns="http://www.w3.org/2000/svg"
			width="48"
			height="48"
			class="bi bi-imag text-slate-500 fill-current"
			viewBox="0 0 20 20"
		>
			<path d="M6.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z" />
			<path
				d="M2.002 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2h-12zm12 1a1 1 0 0 1 1 1v6.5l-3.777-1.947a.5.5 0 0 0-.577.093l-3.71 3.71-2.66-1.772a.5.5 0 0 0-.63.062L1.002 12V3a1 1 0 0 1 1-1h12z"
			/>
		</svg>
	</div>
</div>
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

<script lang="ts">
    import type { DataPanelRows, ColumnInfo, DataPanelSchema } from '$lib/api/datapanel';
	import { Slider } from 'carbon-components-svelte';
	import Card from './Card.svelte';
	import InfoModal from './InfoModal.svelte';

	export let schema: DataPanelSchema;
    let column_infos: Array<ColumnInfo> = schema.columns; 
    export let rows: DataPanelRows | null;

	export let layout = 'masonry'; // 'gimages' or 'masonry'
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
</script>

<svelte:head>
	<link rel="stylesheet" href="https://unpkg.com/carbon-components-svelte/css/g90.css" />
</svelte:head>

<div class="h-full overflow-hidden">
	<div class="m-2">
		<Slider
			fullWidth
			labelText={layout === 'gimages' ? 'Image height' : 'Image columns'}
			min={layout === 'gimages' ? 5 : 1}
			max={layout === 'gimages' ? 95 : 12}
			minLabel={layout === 'gimages' ? '5%' : '1'}
			maxLabel={layout === 'gimages' ? '95%' : '12'}
			bind:value={cell_size_variable}
		/>
	</div>
	<div
		class="panel overflow-y-auto"
		class:panel-masonry={layout === 'masonry'}
		class:panel-gimages={layout === 'gimages'}
		style:height={layout === 'gimages' ? '720px' : 'auto'}
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
			>
				<div slot="pivot-tooltip">Click to see example</div>
			</Card>
		{/each}
	</div>
</div>

<style>
	.panel {
		@apply dark:bg-slate-600;
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

<script lang="ts">
	import type { Writable } from 'svelte/store';
	import { getContext } from 'svelte';
	import ScatterPlot from '$lib/components/plot/layercake/ScatterPlot.svelte';
	import HorizontalBarPlot from './bar/HorizontalBarPlot.svelte';
	import FancyHorizontalBarPlot from './bar/FancyHorizontalBarPlot.svelte';
	import type { Point2D } from '$lib/components/plot/types';

	const { get_rows, remove_row_by_index } = getContext('Interface');

	export let dp: Writable;
	export let selection: Writable;
	export let x: Writable;
	export let y: Writable;
	export let id: string;
	export let x_label: Writable;
	export let y_label: Writable;
	export let type: string;
	export let padding: number = 10;
	export let keys_to_remove: Writable;
	export let can_remove: boolean = true;

	// The columns corresponding to metadata to track.
	export let metadata_columns: Array<string> = [];
	// Array of metadata objects. Each metadata object can have any arbitrary number
	// of key-value pairs.
	let metadata: Array<any> = [];
	
	console.log(metadata_columns)
	let get_datum = async (box_id: string): Promise<Array<Point2D>> => {
		// Fetch all the data from the datapanel for the columns to be plotted
		let rows = await $get_rows(box_id, 0, undefined, undefined, [$x, $y, id, ...metadata_columns]);
		let datum: Array<Point2D> = [];
		rows.rows?.forEach((row: any, index: number) => {
			datum.push({
				x: parseFloat(row[0]),
				y: row[1],
				id: rows.indices[index],
				key: row[2]
			});
		});

		// Update the metadata array.
		if (metadata_columns.length > 0) {
			metadata = [];
			rows.rows?.forEach((row: any) => {
				const metadata_obj = metadata_columns.reduce((accumulator: any, column: string, index: number) => {
					accumulator[index] = row[index + 3];
					return accumulator;
				}, {});
				console.log(metadata_obj)
				metadata.push(metadata_obj);
			});
		} else {
			metadata = new Array(datum.length).fill([]);
		}

		return datum;
	};
	$: datum_promise = get_datum($dp.box_id);
</script>

<!-- TODO: Figure out the padding to put here.  -->
<div class="flex-1 flex flex-col items-center ml-16">
	{#await datum_promise}
		<ScatterPlot
			data={[{ x: 0, y: 0, id: 0, key: 0 }]}
			metadata={[]}
			bind:xlabel={$x_label}
			bind:ylabel={$y_label}
			width="90%"
			height="300px"
		/>
	{:then datum}
		<FancyHorizontalBarPlot
			data={datum}
			metadata={metadata}
			bind:xlabel={$x_label}
			bind:ylabel={$y_label}
			ywidth={128}
			{padding}
			{can_remove}
			on:selection-change={(e) => {
				$selection = Array.from(e.detail.selected_points);
			}}
			on:remove={async (e) => {
				$keys_to_remove.push(datum[e.detail].key);
				$keys_to_remove = $keys_to_remove;
			}}
		/>
	{/await}
</div>

<script lang="ts">
	import type { Writable } from 'svelte/store';
	import { getContext, setContext } from 'svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import ScatterPlot from '$lib/shared/plot/layercake/ScatterPlot.svelte';
	import HorizontalBarPlot from './bar/HorizontalBarPlot.svelte';
	import FancyHorizontalBarPlot from './bar/FancyHorizontalBarPlot.svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import type { Point2D } from '$lib/shared/plot/types';
	import type { Endpoint } from '$lib/utils/types';
	import type { DataFrameChunk, DataFrameRef } from '$lib/api/dataframe';

	const { fetch_chunk, fetch_schema, dispatch } = getContext('Meerkat');

	export let df: DataFrameRef;
	export let x: string;
	export let y: string;
	export let x_label: string;
	export let y_label: string;
	export let type: string;
	export let padding: number = 10;
	export let keys_to_remove: Array<string>;
	export let can_remove: boolean = true;
	export let on_select: Endpoint = null;
	export let on_remove: Endpoint;



	// The columns corresponding to metadata to track.
	export let metadata_columns: Array<string>;

	// Array of metadata objects. Each metadata object can have any arbitrary number
	// of key-value pairs.
	let metadata: Array<any> = [];

	let page: number = 0;
	let per_page: number = 30;


	setContext('removeRow', (id: any) => {
		dispatch(on_remove.endpoint_id, { detail: { slice_id: id } });
	});

	$: schema_promise = fetch_schema({ df: df });

	let get_datum = async (
		df: DataFrameRef,
		page: number,
		per_page: number
	): Promise<Array<Point2D>> => {
		// Fetch all the data from the dataframe for the columns to be plotted
		let chunk: DataFrameChunk = await fetch_chunk({
			df: df,
			start: page * per_page,
			end: (page + 1) * per_page,
			columns: [x, y, ...metadata_columns]
		});
		// iterate over the rows
		let datum: Array<Point2D> = [];
		for (let i = 0; i < chunk.length(); i++) {
			datum.push({
				x: parseFloat(chunk.get_cell(i, x).data),
				y: chunk.get_cell(i, y).data,
				id: chunk.get_cell(i, chunk.primary_key).data
			});
		}

		// Update the metadata array.
		if (metadata_columns.length > 0) {
			metadata = chunk.rows?.map((row: any, i: number) => {
				return metadata_columns.map((column: string) => chunk.get_cell(i, column).data);
			});
		} else {
			metadata = new Array(datum.length).fill([]);
		}

		return datum;
	};
	$: datum_promise = get_datum(df, page, per_page);

	let on_select_run = async (slice_ids: Array<any>) => {
		if (on_select === null) {
			return;
		}

		const promise = dispatch(on_select.endpoint_id, {
			detail: {
				// FIXME: Should we support multiple selections?
				// If there is nothing in the array we should return an empty string
				slice_id: slice_ids.length > 0 ? slice_ids[0] : ''
			}
		});
		promise.catch((error: TypeError) => {
			console.log(error);
		});
	};
</script>

<!-- TODO: Figure out the padding to put here.  -->
<div class="flex-1 flex flex-col items-center w-full">
	{#await datum_promise}
		<div class="flex justify-center items-center h-full">
			<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
		</div>
	{:then datum}
		<FancyHorizontalBarPlot
			data={datum}
			{metadata}
			bind:xlabel={x_label}
			bind:ylabel={y_label}
			ywidth={196}
			{padding}
			{can_remove}
			on:selection-change={(e) => {
				on_select_run(Array.from(e.detail.selected_points));
			}}
			on:remove={async (e) => {
				keys_to_remove.push(datum[e.detail].key);
				keys_to_remove = keys_to_remove;
			}}
		/>
	{/await}
	{#await schema_promise then schema}
		<div class=" z-10 bottom-0 w-full m-0 py-3">
			<Pagination bind:page bind:per_page loaded_items={schema.nrows} total_items={schema.nrows} />
		</div>
	{/await}
</div>

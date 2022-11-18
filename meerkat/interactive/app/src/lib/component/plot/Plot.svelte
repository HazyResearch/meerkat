<script lang="ts">
	import type { Writable } from 'svelte/store';
	import { getContext } from 'svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import ScatterPlot from '$lib/shared/plot/layercake/ScatterPlot.svelte';
	import HorizontalBarPlot from './bar/HorizontalBarPlot.svelte';
	import FancyHorizontalBarPlot from './bar/FancyHorizontalBarPlot.svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import type { Point2D } from '$lib/shared/plot/types';
	import type { Endpoint } from '$lib/utils/types';

	const { get_rows, remove_row_by_index, get_schema, dispatch } = getContext('Interface');

	export let df: Writable;
	export let x: Writable<string>;
	export let y: Writable<string>;
	export let primary_key: Writable<string>;
	export let x_label: Writable<string>;
	export let y_label: Writable<string>;
	export let type: Writable<string>;
	export let padding: number = 10;
	export let keys_to_remove: Writable;
	export let can_remove: boolean = true;
	export let on_select: Endpoint = null;

	// The columns corresponding to metadata to track.
	export let metadata_columns: Writable<Array<string>>;

	// Array of metadata objects. Each metadata object can have any arbitrary number
	// of key-value pairs.
	let metadata: Array<any> = [];

	let page: number = 0;
	let per_page: number = 30;

	$: schema_promise = $get_schema($df.ref_id);

	let get_datum = async (
		ref_id: string,
		page: number,
		per_page: number
	): Promise<Array<Point2D>> => {
		// Fetch all the data from the dataframe for the columns to be plotted
		let rows = await $get_rows(ref_id, page * per_page, (page + 1) * per_page, undefined, [
			$x,
			$y,
			$primary_key,
			...$metadata_columns
		]);
		let datum: Array<Point2D> = [];
		rows.rows?.forEach((row: any, index: number) => {
			datum.push({
				x: parseFloat(row[0]),
				y: row[1],
				id: row[2]
			});
		});

		// Update the metadata array.
		if ($metadata_columns.length > 0) {
			metadata = [];
			rows.rows?.forEach((row: any) => {
				const metadata_obj = $metadata_columns.reduce(
					(accumulator: any, column: string, index: number) => {
						accumulator[index] = row[index + 3];
						return accumulator;
					},
					{}
				);
				metadata.push(metadata_obj);
			});
		} else {
			metadata = new Array(datum.length).fill([]);
		}

		return datum;
	};
	$: datum_promise = get_datum($df.ref_id, page, per_page);

	let on_select_run = async (slice_ids: Array<any>) => {
		if (on_select === null) {
			return;
		}

		status = 'working';
		const promise = $dispatch(
			on_select.endpoint_id,
			{
				// FIXME: Should we support multiple selections?
				// If there is nothing in the array we should return an empty string
				slice_id: slice_ids.length > 0 ? slice_ids[0] : ''
			},
			{}
		);
		promise
			.then(() => {
				status = 'success';
			})
			.catch((error: TypeError) => {
				status = 'error';
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
			bind:xlabel={$x_label}
			bind:ylabel={$y_label}
			ywidth={196}
			{padding}
			{can_remove}
			on:selection-change={(e) => {
				on_select_run(Array.from(e.detail.selected_points));
			}}
			on:remove={async (e) => {
				$keys_to_remove.push(datum[e.detail].key);
				$keys_to_remove = $keys_to_remove;
			}}
		/>
	{/await}
	{#await schema_promise then schema}
		<div class=" z-10 bottom-0 w-full m-0 py-3">
			<Pagination bind:page bind:per_page loaded_items={schema.nrows} total_items={schema.nrows} />
		</div>
	{/await}
</div>

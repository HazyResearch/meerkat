<script lang="ts">
	import { api_url } from '$lib/../routes/network/stores';
	import { fetch_chunk } from '$lib/api/api';
	import {
		MatchCriterion,
		type DataFrameRows,
		type DataFrameSchema
	} from '$lib/api/dataframe';
	import type { RefreshCallback } from '$lib/shared/deprecate/callbacks';
	import MatchHeader from '$lib/shared/deprecate/match_header/MatchHeader.svelte';
	import ScatterPlot from '$lib/shared/plot/layercake/ScatterPlot.svelte';
	import type { Point2D } from '../../plot/types';
	import ColumnSelect from '../column_select/ColumnSelect.svelte';
	import Toggle from '../common/Toggle.svelte';

	export let dataframe_id: string;
	export let rows_promise = Promise<DataFrameRows>;
	export let schema_promise: Promise<DataFrameSchema>;
	export let refresh_callback: RefreshCallback;
	export let match_criterion: MatchCriterion;
	export let x_match_interface = false;
	export let y_match_interface = false;

	export let xlabel: string = '';
	export let xcolumn: string = '';
	export let ylabel: string = '';
	export let ycolumn: string = '';
	$: imputed_xcolumn = x_match_interface ? `_match_${xcolumn}_${xlabel}` : xcolumn;
	$: imputed_ycolumn = y_match_interface ? `_match_${ycolumn}_${ylabel}` : ycolumn;

	$: {
		if (!x_match_interface) xlabel = xcolumn;
		if (!y_match_interface) ylabel = ycolumn;
	}

	let get_datum = async (dataframe_id: string): Promise<Array<Point2D>> => {
		// Fetch all the data from the dataframe for the columns to be plotted
		console.log(`Fetching data for dataframe ${dataframe_id}`);
		let imputed_xcolumn = x_match_interface ? `_match_${xcolumn}_${xlabel}` : xcolumn;
		let imputed_ycolumn = y_match_interface ? `_match_${ycolumn}_${ylabel}` : ycolumn;

		console.log(imputed_xcolumn, imputed_ycolumn);

		let rows = await fetch_chunk($api_url, dataframe_id, 0, undefined, undefined, [
			imputed_xcolumn,
			imputed_ycolumn
		]);
		let datum: Array<Point2D> = [];
		rows.rows?.forEach((row: any, index: number) => {
			datum.push({
				x: parseFloat(row[0]),
				y: parseFloat(row[1]),
				id: rows.indices[index]
			});
		});
		return datum;
	};
	$: datum_promise = get_datum(dataframe_id);
</script>

<div>
	<div class="flex flex-col items-center mb-4">
		{#await datum_promise}
			<ScatterPlot
				data={[{ x: 0, y: 0, id: 0 }]}
				bind:xlabel
				bind:ylabel
				width="90%"
				height="300px"
			/>
		{:then datum}
			<ScatterPlot
				data={datum}
				bind:xlabel
				bind:ylabel
				width="90%"
				height="300px"
				on:selection-change
			/>
		{/await}
	</div>

	<div class="m-2" />
	<Toggle label_left="Direct" label_right="Match" bind:checked={x_match_interface} />
	<div class="m-2" />
	{#if x_match_interface}
		<div class="text-sm text-slate-200 font-mono">X Label against {xcolumn}</div>
		<MatchHeader
			{schema_promise}
			{refresh_callback}
			bind:match_criterion
			bind:search_box_text={xlabel}
			bind:column={xcolumn}
			on:match={(event) => {
				datum_promise = get_datum(dataframe_id);
			}}
		/>
	{:else}
		<div class="flex justify-center items-center">
			<div class="text-sm text-slate-200 font-mono">X Label</div>
			<ColumnSelect
				{schema_promise}
				bind:column={xcolumn}
				on:select={(event) => {
					datum_promise = get_datum(dataframe_id);
				}}
			/>
		</div>
	{/if}

	<div class="m-2" />
	<Toggle label_left="Direct" label_right="Match" bind:checked={y_match_interface} />
	<div class="m-2" />
	{#if y_match_interface}
		<div class="text-sm text-slate-200 font-mono">Y Label against {ycolumn}</div>
		<MatchHeader
			{schema_promise}
			{refresh_callback}
			bind:match_criterion
			bind:search_box_text={ylabel}
			bind:column={ycolumn}
			on:match={(event) => {
				datum_promise = get_datum(dataframe_id);
			}}
		/>
	{:else}
		<div class="flex justify-center items-center">
			<div class="text-sm text-slate-200 font-mono">Y Label</div>
			<!-- TODO: ColumnSelect should only have columns that are plottable (e.g. no image columns) -->
			<ColumnSelect
				{schema_promise}
				bind:column={ycolumn}
				on:select={(event) => {
					datum_promise = get_datum(dataframe_id);
				}}
			/>
		</div>
	{/if}
</div>

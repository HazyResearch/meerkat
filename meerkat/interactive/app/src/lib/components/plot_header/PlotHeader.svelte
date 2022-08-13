<script lang="ts">
	import { api_url } from '$lib/../routes/network/stores';
	import type { RefreshCallback } from '$lib/api/callbacks';
	import { get_rows, type DataPanelRows, type DataPanelSchema } from '$lib/api/datapanel';
	import MatchHeader from '$lib/components/match_header/MatchHeader.svelte';
	import ScatterPlot from '$lib/components/plot/layercake/ScatterPlot.svelte';
	import type { Point2D } from '../plot/types';

	export let datapanel_id: string;
	export let rows_promise = Promise<DataPanelRows>;
	export let schema_promise: Promise<DataPanelSchema>;
	export let refresh_callback: RefreshCallback;

	export let xlabel: string = 'blue';
	export let xcolumn: string = 'img';
	export let ylabel: string = 'red';
	export let ycolumn: string = 'img';
	$: imputed_xcolumn = `_match_${xcolumn}_${xlabel}`;
	$: imputed_ycolumn = `_match_${ycolumn}_${ylabel}`;

	let get_datum = async (datapanel_id: string): Promise<Array<Point2D>> => {
        // Fetch all the data from the datapanel for the columns to be plotted
		let rows = await get_rows($api_url, datapanel_id, 0, undefined, undefined, [
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
	$: datum_promise = get_datum(datapanel_id);
</script>

<div>
	<div class="flex flex-col items-center mb-4">
		{#await datum_promise}
			<ScatterPlot data={[]} bind:xlabel bind:ylabel width="90%" height="300px" />
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

	<div class="text-sm text-slate-200 font-mono">X Label against {xcolumn}</div>
	<MatchHeader
		base_datapanel_id={datapanel_id}
		{schema_promise}
		{refresh_callback}
		bind:search_box_text={xlabel}
		bind:column={xcolumn}
		on:match={(event) => { datum_promise = get_datum(datapanel_id); }}
	/>
	<div class="text-sm text-slate-200 font-mono">Y Label against {ycolumn}</div>
	<MatchHeader
		base_datapanel_id={datapanel_id}
		{schema_promise}
		{refresh_callback}
		bind:search_box_text={ylabel}
		bind:column={ycolumn}
		on:match={(event) => { datum_promise = get_datum(datapanel_id); }}
	/>
</div>

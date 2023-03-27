<script lang="ts">
	import { dispatch, fetchChunk } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import Plot from '../plot/Plot.svelte';

	export let df: DataFrameRef;
	export let x: string;
	export let y: string;
	export let orientation: string = 'v';
	export let config: Record<string, any> = { displayModeBar: false };
	export let title: string;
	export let onClick: Endpoint;

	$: data_promise = fetchChunk({
		df: df,
		start: 0,
		columns: [x, y],
		variants: ['small']
	}).then((chunk) => {
		return [
			{
				x: chunk.getColumn(x).data,
				y: chunk.getColumn(y).data,
				keyidx: chunk.getColumn(chunk.primaryKey).data,
				type: 'bar',
				orientation: orientation
			}
		];
	});

	const layout = { xaxis: { title: x }, yaxis: { title: y }, title: title };

	async function on_endpoint(endpoint: Endpoint, e) {
		let data = await data_promise;
		e.detail.points;
		if (endpoint) {
			dispatch(endpoint.endpointId, {
				detail: {
					keyidxs: e.detail.points.map((p) => data[0].keyidx[p.pointIndex])
				}
			});
		}
	}
</script>

{#await data_promise}
    <div class="bg-purple-50 text-center my-1">Loading Bar Plot...</div>
    <Plot data={[]} {layout} {config} on:click={(e) => on_endpoint(onClick, e)} />
{:then data}
	<Plot {data} {layout} {config} on:click={(e) => on_endpoint(onClick, e)} />
{/await}

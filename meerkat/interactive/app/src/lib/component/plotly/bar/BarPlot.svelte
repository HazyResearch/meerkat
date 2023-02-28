<script lang="ts">
	import { dispatch, fetchChunk } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import Plot from '../plot/Plot.svelte';

	export let df: DataFrameRef;
	export let x: string;
	export let y: string;
	export let config: Record<string, any> = { displayModeBar: false };
	export let title: string;
	export let on_click: Endpoint;

	$: data_promise = fetchChunk({
		df: df,
		start: 0,
		columns: [x, y],
		variants: ['small']
	}).then((chunk) => {
		console.log(chunk.primaryKey)
		return [
			{
				x: chunk.getColumn(x).data,
				y: chunk.getColumn(y).data,
				keyidx: chunk.getColumn(chunk.primaryKey).data,
				type: 'bar',
			}
		];
	});

	const layout = { xaxis: { title: x }, yaxis: { title: y }, title: title };

	async function on_endpoint(endpoint: Endpoint, e) {
		let data = await data_promise;
		e.detail.points;
		console.log(e);
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
    <Plot data={[]} {layout} {config} on:click={(e) => on_endpoint(on_click, e)} />
{:then data}
	<Plot {data} {layout} {config} on:click={(e) => on_endpoint(on_click, e)} />
{/await}

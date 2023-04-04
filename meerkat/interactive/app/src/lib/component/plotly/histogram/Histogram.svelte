<script lang="ts">
	import { dispatch, fetchChunk } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import Plot from '../plot/Plot.svelte';

	export let df: DataFrameRef;
	export let x: string;
	export let config: Record<string, any> = { displayModeBar: false };
	export let color: string;
	export let title: string;
	export let nbins: number;
	export let on_click: Endpoint;

	console.log('columns: ', [x, color]);

	$: data_promise = fetchChunk({
		df: df,
		start: 0,
		columns: [x, color],
		variants: ['small']
	}).then((chunk) => {
		return [
			{
				x: chunk.getColumn(x).data,
				color: chunk.getColumn(color).data,
				keyidx: chunk.getColumn(chunk.primaryKey).data,
				type: 'histogram'
			}
		];
	});

	const layout = {
		xaxis: { title: x },
		yaxis: { title: 'count' },
		title: title,
		nbins: nbins,
		color: color
	};

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

{#await data_promise then data}
	<Plot {data} {layout} {config} on:click={(e) => on_endpoint(on_click, e)} />
{/await}

<script lang="ts">
	import { dispatch, fetchChunk } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import Plot from '../plot/Plot.svelte';

	export let df: DataFrameRef;
	export let lat: string;
	export let lon: string;
	export let config: Record<string, any> = { displayModeBar: false };
	export let title: string;
	export let on_click: Endpoint;

	$: data_promise = fetchChunk({
		df: df,
		start: 0,
		columns: [lat, lon],
		variants: ['small']
	}).then((chunk) => {
		return [
			{
				lat: chunk.getColumn(lat).data,
				lon: chunk.getColumn(lon).data,
				keyidx: chunk.getColumn(chunk.primaryKey).data,
				mode: 'markers',
				marker: {
					size: 14
				},
				type: 'scattermapbox'
			}
		];
	});

	const layout = {
		title: title,
		autosize: true,
		// hovermode: 'closest',
		mapbox: {
			// bearing: 0,
			center: {
				lat: 47.6,
				lon: -122.3
			},
			// pitch: 0,
			// zoom: 8,
			// width: 800,
			// height: 600
		}
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

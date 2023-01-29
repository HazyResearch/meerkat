<script lang="ts">
	import Plot from '../plot/Plot.svelte'
	import { getContext } from 'svelte';
	import type { DataFrameRef } from '$lib/api/dataframe';
	import type { Endpoint } from '$lib/utils/types';

	const { fetch_chunk, dispatch } = getContext('Meerkat');

	export let df: DataFrameRef;
	export let x: string;
	export let y: string;
    export let config: Record<string, any>  = {displayModeBar: false}
    export let on_click: Endpoint;

	$: data_promise = fetch_chunk({
		df: df,
		start: 0,
		columns: [x, y],
		variants: ['small']
	}).then((chunk) => {
		return [{
			x: chunk.get_column(x).data,
			y: chunk.get_column(y).data,
            keyidx: chunk.get_column(chunk.primary_key).data,
			type: 'bar'
		}];
	});

    const layout = {xaxis: { title: x}, yaxis: {title: y}}

    async function on_endpoint(endpoint:Endpoint, e){
        let data = await data_promise
        e.detail.points
        console.log(e);
        if (endpoint) {
            dispatch(endpoint.endpoint_id, {"detail": {
                keyidxs: e.detail.points.map((p) => data[0].keyidx[p.pointIndex]),
            }});
        }
    }
</script>

{#await data_promise then data}
	<Plot {data} {layout} {config} on:click={(e) => on_endpoint(on_click, e) }/>
{/await}

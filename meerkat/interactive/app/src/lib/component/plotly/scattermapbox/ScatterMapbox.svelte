<script lang="ts">
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import Plot from '../plot/Plot.svelte';

	export let json_desc: string;
	export let on_click: Endpoint;

	console.log("json_desc yo:", json_desc);

	async function on_endpoint(endpoint: Endpoint, e) {
		let d = JSON.parse(json_desc).data;
		e.detail.points;
		console.log(e);
		if (endpoint) {
			dispatch(endpoint.endpointId, {
				detail: {
					keyidxs: e.detail.points.map((p) => d[0].keyidx[p.pointIndex])
				}
			});
		}
	}
</script>

<Plot data={JSON.parse(json_desc).data} on:click={(e) => on_endpoint(on_click, e)} />

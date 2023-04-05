<script lang="ts">
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import Plot from '../plot/Plot.svelte';

	export let jsonDesc: string;
	export let onClick: Endpoint;

	console.log("jsonDesc yo:", jsonDesc);

	async function on_endpoint(endpoint: Endpoint, e) {
		let d = JSON.parse(jsonDesc).data;
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

<Plot data={JSON.parse(jsonDesc).data} on:click={(e) => on_endpoint(onClick, e)} />

<script lang="ts">
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import Plot from '../plot/Plot.svelte';

	export let jsonDesc: string;
	export let onClick: Endpoint;

	async function onEndpoint(endpoint: Endpoint, e) {
		let data = JSON.parse(jsonDesc).data;
		console.log(e.detail.points);
		console.log(data);
		if (endpoint) {
			dispatch(endpoint.endpointId, {
				detail: {
					keyidxs: e.detail.points.map((p) => data[0].customdata[p.pointIndex])
				}
			});
		}
	}
</script>

<Plot {...JSON.parse(jsonDesc)} on:click={(e) => onEndpoint(onClick, e)} />

<script lang="ts">
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import { createEventDispatcher } from 'svelte';
	import Plot from '../plot/Plot.svelte';

	const eventDispatcher = createEventDispatcher();

	export let jsonDesc: string;
	export let onClick: Endpoint;
	export let selected: Array<number> = [];

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

	async function onSelected(e) {
		if (e.detail) {
			selected = e.detail.points.map((p) =>
				p.data.keyidx ? p.data.keyidx[p.pointIndex] : p.pointIndex
			);
		} else {
			selected = [];
		}
		eventDispatcher('select', { selected: selected });
	}
</script>

<Plot
	{...JSON.parse(jsonDesc)}
	on:click={(e) => onEndpoint(onClick, e)}
	on:selected={(e) => onSelected(e)}
/>

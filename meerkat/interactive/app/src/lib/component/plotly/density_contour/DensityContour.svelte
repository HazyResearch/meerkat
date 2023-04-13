<script lang="ts">
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import { createEventDispatcher } from 'svelte';
	import Plot from '../plot/Plot.svelte';

	const eventDispatcher = createEventDispatcher();

	export let keyidxs: Array<string | number>;

	export let jsonDesc: string;
	export let onClick: Endpoint;
	export let selected: Array<number> = [];

	async function onEndpoint(endpoint: Endpoint, e) {
		if (!endpoint) return;
		console.log('endpoint', e.detail)
		dispatch(endpoint.endpointId, {
			detail: { keyidxs: e.detail.points.map((p) => keyidxs[p.pointIndex]) }
		});
	}

	async function onSelected(e) {
		selected = e.detail ? e.detail.points.map((p) => keyidxs[p.pointIndex]) : [];
		eventDispatcher('select', { selected: selected });
	}
</script>

<Plot
	{...JSON.parse(jsonDesc)}
	on:click={(e) => onEndpoint(onClick, e)}
	on:selected={(e) => onSelected(e)}
/>

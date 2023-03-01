<script lang="ts">
	import { dispatch, fetchChunk } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import { createEventDispatcher } from 'svelte';
	import Plot from '../plot/Plot.svelte';

	const eventDispatcher = createEventDispatcher();

	export let df: DataFrameRef;
	export let x: string;
	export let y: string;
	export let hue: string | null = null;
	export let config: Record<string, any> = { displayModeBar: false };
	export let title: string;
	export let selected: Array<number> = [];
	export let buffer: boolean = false;
	export let onClick: Endpoint;

	$: dataPromise = fetchChunk({
		df: df,
		start: 0,
		end: buffer ? 1000 : null,
		columns: [x, y].concat(hue ? [hue] : []),
		variants: ['small'],
		shuffle: buffer
	}).then((chunk) => {
		let categories = null;
		if (hue !== null) {
			// Get the unique categories from the hue column
			categories = [...new Set(chunk.getColumn(hue).data)];
			const booleanCategory = categories.every((c) => typeof c === 'boolean');
			console.log(categories);
			return categories.map((category) => {
				return {
					x: chunk.getColumn(x).data.filter((_, i) => chunk.getColumn(hue).data[i] === category),
					y: chunk.getColumn(y).data.filter((_, i) => chunk.getColumn(hue).data[i] === category),
					keyidx: chunk
						.getColumn(chunk.primaryKey)
						.data.filter((_, i) => chunk.getColumn(hue).data[i] === category),
					type: 'scatter',
					mode: 'markers',
					name: booleanCategory ? (category === true ? hue : `not ${hue}`) : category
				};
			});
		} else {
			return [
				{
					x: chunk.getColumn(x).data,
					y: chunk.getColumn(y).data,
					keyidx: chunk.getColumn(chunk.primaryKey).data,
					type: 'scatter',
					mode: 'markers'
				}
			];
		}
	});

	$: layout = {
		xaxis: { title: x, categoryorder: 'category ascending', automargin: true },
		yaxis: { title: y, categoryorder: 'category ascending', automargin: true },
		title: title,
		dragmode: 'select'
	};

	async function onEndpoint(endpoint: Endpoint, e) {
		let data = await dataPromise;
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

{#await dataPromise}
	{#if buffer}
		<!-- <div class="bg-purple-50 text-center my-1">Loading Scatter Plot...</div> -->
		<Plot data={[]} {layout} {config} on:click={(e) => onEndpoint(onClick, e)} />
	{:else}
		<svelte:self {df} {x} {y} {hue} {config} {title} on_click={onClick} buffer={true} on:select />
	{/if}
{:then data}
	<Plot
		{data}
		{layout}
		{config}
		on:click={(e) => onEndpoint(onClick, e)}
		on:selected={(e) => {
			if (e.detail) {
				selected = e.detail.points.map((p) => p.data.keyidx[p.pointIndex]);
			} else {
				selected = [];
			}
			eventDispatcher('select', { selected: selected });
		}}
	/>
{/await}

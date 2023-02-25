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
	export let config: Record<string, any> = { displayModeBar: false };
	export let title: string;
	export let selected: Array<number> = [];
	export let buffer: boolean = false;
	export let onClick: Endpoint;

	$: dataPromise = fetchChunk({
		df: df,
		start: 0,
		end: buffer ? 1000 : null,
		columns: [x, y],
		variants: ['small'],
		shuffle: buffer
	}).then((chunk) => {
		return [
			{
				x: chunk.getColumn(x).data,
				y: chunk.getColumn(y).data,
				keyidx: chunk.getColumn(chunk.primaryKey).data,
				type: 'scatter',
				mode: 'markers'
			}
		];
	});

	$: layout = { xaxis: { title: x }, yaxis: { title: y }, title: title, dragmode: 'select' };

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
		<div class="bg-purple-50 text-center my-1">Loading Scatter Plot...</div>
		<Plot data={[]} {layout} {config} on:click={(e) => onEndpoint(onClick, e)} />
	{:else}
		<svelte:self {df} {x} {y} {config} {title} on_click={onClick} buffer={true} on:select />
	{/if}
{:then data}
	<!-- <div class="flex flex-row align-middle justify-center mb-4 bg-transparent">
		<div> -->
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
		<!-- </div>
	</div> -->
{/await}

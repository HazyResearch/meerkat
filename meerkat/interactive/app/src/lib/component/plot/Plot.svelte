<script lang="ts">
	import type { Writable } from 'svelte/store';
	import { getContext } from 'svelte';
	import ScatterPlot from '$lib/components/plot/layercake/ScatterPlot.svelte';
	import BarPlot from './bar/BarPlot.svelte';
	import type { Point2D } from '$lib/components/plot/types';

	const { get_rows } = getContext('Interface');

	export let dp: Writable;
	export let selection: Writable;
	export let x: Writable;
	export let y: Writable;
    export let x_label: Writable;
    export let y_label: Writable;
	export let type: string;

	let get_datum = async (box_id: string): Promise<Array<Point2D>> => {
		// Fetch all the data from the datapanel for the columns to be plotted
		let rows = await $get_rows(box_id, 0, undefined, undefined, [$x, $y]);
		let datum: Array<Point2D> = [];
		rows.rows?.forEach((row: any, index: number) => {
			datum.push({
				x: parseFloat(row[0]),
				y: row[1],
				id: rows.indices[index]
			});
		});
		return datum;
	};
	$: datum_promise = get_datum($dp.box_id);

</script>

<div class="flex-1 flex flex-col items-center pb-8">
	{#await datum_promise}
		<ScatterPlot
			data={[{ x: 0, y: 0, id: 0 }]}
			bind:xlabel={$x_label}
			bind:ylabel={$y_label}
			width="90%"
			height="300px"
		/>
	{:then datum}
		<BarPlot
			data={datum}
			bind:xlabel={$x_label}
			bind:ylabel={$y_label}
			width="90%"
			height="300px"
			on:selection-change={(e) => {
				$selection = Array.from(e.detail.selected_points);
			}}
		/>
	{/await}
</div>

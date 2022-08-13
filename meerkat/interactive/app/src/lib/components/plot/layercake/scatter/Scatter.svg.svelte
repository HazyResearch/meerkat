<script lang="ts">
	import { getContext } from 'svelte';
	import Brush from '../interaction/Brush.svg.svelte';
	import Quadtree from '../utils/Quadtree.svelte';

	const { data, xGet, yGet, xScale, yScale } = getContext('LayerCake');

	export let scanned_points = new Set();
	export let selected_points = new Set();
	export let radius = 5;

	// Bind to Quadtree component (d3 quadtree)
	let quadtree: any;

	// Event handler for brush events
	export let brushed = function (event: any) {
		scanned_points.clear();
		if (!event.sourceEvent.ctrlKey) {
			selected_points.clear();
		}
		if (event.selection) {
			let [_selected, _scanned] = quadtree.search(event.selection);
			selected_points = new Set([...selected_points, ..._selected]);
			scanned_points = new Set([...scanned_points, ..._scanned]);
		}
		[selected_points, scanned_points] = [selected_points, scanned_points]; // trigger re-render
	}

	// Selected scatter points
	let manual_selected_points = new Set();
	let select_point = (id: number) => (e: MouseEvent) => {
		if (manual_selected_points.has(id)) {
			manual_selected_points.delete(id);
		} else {
			manual_selected_points.add(id);
		}
		manual_selected_points = manual_selected_points; // trigger re-render
	};

	// Event handler to clear manually selected points
	let clear_points = (e: MouseEvent) => {
		if (e.ctrlKey) return; // this event fires on MacOS if ctrl + left click is used
		manual_selected_points.clear();
		manual_selected_points = manual_selected_points; // trigger re-render
	};
</script>

<g class="w-full h-full" on:contextmenu={clear_points}>
	<Quadtree bind:this={quadtree} />
	<Brush {brushed} />
	<g class="scatter-group">
		{#each $data as d}
			<circle
				class={selected_points.has(d.id) || manual_selected_points.has(d.id)
					? `scatter-point--selected`
					: scanned_points.has(d.id)
					? `scatter-point--scanned`
					: `scatter-point`}
				cx={$xGet(d) + ($xScale.bandwidth ? $xScale.bandwidth() / 2 : 0)}
				cy={$yGet(d) + ($yScale.bandwidth ? $yScale.bandwidth() / 2 : 0)}
				r={radius}
				on:click={select_point(d.id)}
			/>
		{/each}
	</g>
</g>

<style>
	.scatter-point {
		fill: violet;
		fill-opacity: 0.5;
		stroke: violet;
		stroke-width: 1px;
	}

	.scatter-point:hover {
		fill: orange;
		fill-opacity: 0.5;
		stroke: orange;
		stroke-width: 1px;
	}

	.scatter-point--scanned {
		fill: orange;
		fill-opacity: 0.5;
		stroke: orange;
		stroke-width: 1px;
	}

	.scatter-point--selected {
		fill: red;
		fill-opacity: 0.5;
		stroke: red;
		stroke-width: 3px;
	}
</style>

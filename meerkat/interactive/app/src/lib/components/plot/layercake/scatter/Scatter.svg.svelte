<script lang="ts">
	import { getContext } from 'svelte';
	import Brush from '../interaction/Brush.svg.svelte';
	import Quadtree from '../utils/Quadtree.svelte';
	import { createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher();
	const { data, xGet, yGet, xScale, yScale } = getContext('LayerCake');

	export let scanned_points = new Set();
	export let selected_points = new Set();
	export let radius = 5;
	export let show_scanned = false;

	// Bind to Quadtree component (d3 quadtree)
	let quadtree: any;

	function dispatch_selection_change() {
		dispatch('selection-change', {
			selected_points: selected_points
		});
	};

	// Event handler for brush events
	export let brushed = function (event: any) {
		// Clear out the scanned points
		scanned_points.clear();
		// No Ctrl key pressed, so clear out the selected points
		if (!event.sourceEvent.ctrlKey) {
			selected_points.clear();
			[selected_points, scanned_points] = [selected_points, scanned_points]; // trigger re-render
		}
		if (event.selection) {
			let [_selected, _scanned] = quadtree.search(event.selection);
			selected_points = new Set([...selected_points, ..._selected]);
			scanned_points = new Set([...scanned_points, ..._scanned]);
		}
		// Dispatch the selection change event only on end of brushing
		if (event.type === 'end') dispatch_selection_change();
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
	<Brush {brushed} fire_on="start brush end" />
	<g class="scatter-group">
		{#each $data as d}
			<circle
				class={selected_points.has(d.id) || manual_selected_points.has(d.id)
					? `scatter-point--selected`
					: scanned_points.has(d.id)
					? (show_scanned ? `scatter-point--scanned` : `scatter-point`)
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

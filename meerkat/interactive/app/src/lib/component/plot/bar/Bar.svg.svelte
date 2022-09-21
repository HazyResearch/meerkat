<script lang="ts">
	import { getContext } from 'svelte';
	import Brush from '$lib/components/plot/layercake/interaction/Brush.svg.svelte';
	import { createEventDispatcher } from 'svelte';
	import {interpolatePiYG} from "d3-scale-chromatic";
import { interpolate } from 'd3';

	const dispatch = createEventDispatcher();
	const { data, xGet, yGet, xScale, yScale, x } = getContext('LayerCake');

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
	}

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
	};

	// Selected scatter points
	let manual_selected_points = new Set();
	let select_point = (id: number) => (e: MouseEvent) => {
		if (selected_points.has(id)) {
			selected_points.delete(id);
		} else {
			selected_points.add(id);
		}
		selected_points = selected_points; // trigger re-render
        dispatch_selection_change();
	};

	// Event handler to clear manually selected points
	let clear_points = (e: MouseEvent) => {
		if (e.ctrlKey) return; // this event fires on MacOS if ctrl + left click is used
		selected_points.clear();
		selected_points = selected_points; // trigger re-render
	};

</script>


<g class="w-full h-full" on:contextmenu={clear_points}>
	<Brush {brushed} fire_on="start brush end" />
	<g class="bar-group">
		{#each $data as d}
			{@const x = ($x(d) >= 0) ? $xGet({x: 0}) : $xGet(d)}
			{@const width = Math.abs($xGet(d) - $xGet({x: 0}))}

			<rect
				class={selected_points.has(d.id) || manual_selected_points.has(d.id)
					? `scatter-point--selected`
					: scanned_points.has(d.id)
					? show_scanned
						? `scatter-point--scanned`
						: `scatter-point`
					: `scatter-point`}
				x="{x}"
                y="{$yGet(d)}"
                height={$yScale.bandwidth()}
                width="{width}"
				rx=3
				on:click={select_point(d.id)}
				style="fill: {interpolatePiYG((d.x + 1) / 2)};"
			/>
		{/each}
	</g>
</g>

<style>
	.scatter-point {
        @apply fill-slate-400;
		stroke-width: 1px;
	}

	.scatter-point:hover {
        @apply fill-violet-200;
		@apply opacity-50;
		stroke-width: 1px;
	}

	.scatter-point--scanned {
        @apply fill-violet-200;
		stroke-width: 1px;
	}

	.scatter-point--selected {
        @apply fill-violet-600;
		@apply border stroke-violet-600;
		stroke-width: 3px;
	}
</style>

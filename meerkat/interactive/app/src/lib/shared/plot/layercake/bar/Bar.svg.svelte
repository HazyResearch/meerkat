<script lang="ts">
	import { interpolatePuOr } from 'd3-scale-chromatic';
	import { createEventDispatcher, getContext } from 'svelte';

	const dispatch = createEventDispatcher();
	const { data, xGet, yGet, yScale, x } = getContext('LayerCake');

	export let selected_bars = new Set();

	// Allow the user to only select one bar at a time
	export let single_selection = true;

	function dispatch_selection_change() {
		dispatch('selection-change', {
			selected_points: selected_bars
		});
	}

	// Selected scatter points
	let select_point = (id: number) => (e: MouseEvent) => {
		if (selected_bars.has(id)) {
			selected_bars.delete(id);
		} else {
			if (single_selection) {
				selected_bars.clear();
			}
			selected_bars.add(id);
		}
		selected_bars = selected_bars; // trigger re-render
		dispatch_selection_change();
	};

	// Event handler to clear manually selected points
	let clear_points = (e: MouseEvent) => {
		if (e.ctrlKey) return; // this event fires on MacOS if ctrl + left click is used
		selected_bars.clear();
		selected_bars = selected_bars; // trigger re-render
	};
</script>

<g class="w-full" on:contextmenu={clear_points}>
	<g class="bar-group">
		{#each $data as d}
			{@const x = $x(d) >= 0 ? $xGet({ x: 0 }) : $xGet(d)}
			{@const width = Math.abs($xGet(d) - $xGet({ x: 0 }))}

			<rect
				class={selected_bars.has(d.id) ? `bar--selected` : `bar`}
				{x}
				y={$yGet(d)}
				height={$yScale.bandwidth() * 0.95}
				{width}
				rx="3"
				on:click={select_point(d.id)}
			/>
		{/each}
	</g>
</g>

<style>
	.bar {
		@apply fill-slate-400 fill-violet-300;
		stroke-width: 1px;
	}

	.bar:hover {
		@apply fill-violet-200;
		@apply opacity-50;
		stroke-width: 1px;
	}

	.bar--selected {
		@apply fill-violet-600;
		@apply border stroke-violet-600;
		stroke-width: 3px;
	}
</style>

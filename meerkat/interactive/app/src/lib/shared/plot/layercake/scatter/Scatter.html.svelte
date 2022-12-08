<!--
	@component
	Generates an HTML scatter plot. This component can also work if the x- or y-scale is ordinal, i.e. it has a `.bandwidth` method. See the [timeplot chart](https://layercake.graphics/example/Timeplot) for an example.
 -->
<script lang="ts">
	import { getContext } from 'svelte';
	import Styling from '$lib/shared/common/Styling.svelte';

	const { data, xGet, yGet, xScale, yScale, width, height } = getContext('LayerCake');

	export let cssvars = {};
	export let tailwind: string;

	// Selected scatter points
	let selected = new Set();
	let select_point = (index: number) => (e: MouseEvent) => {
		if (selected.has(index)) {
			selected.delete(index);
		} else {
			selected.add(index);
		}
		selected = selected; // trigger re-render
	};
</script>

<Styling {cssvars}>
	<div class="scatter-group">
		{#each $data as d, i}
			<div
				class="scatter-point {tailwind}"
				class:scatter-point-selected={selected.has(i)}
				style:left="{$xGet(d) + ($xScale.bandwidth ? $xScale.bandwidth() / 2 : 0)}%"
				style:top="{$yGet(d) + ($yScale.bandwidth ? $yScale.bandwidth() / 2 : 0)}%"
				on:click={select_point(i)}
			/>
		{/each}
	</div>
</Styling>

<style>
	.scatter-point {
		--radius: 12px;
		--z: 1;
		@apply w-[var(--radius)] h-[var(--radius)];
		@apply absolute -translate-x-1/2 -translate-y-1/2 rounded-full;
		@apply border-2 border-solid border-violet-300;
		@apply bg-gray-500 z-[var(--z)];
	}

	.scatter-point-selected {
		@apply bg-gray-300 border-2 border-solid border-violet-600;
	}

	.scatter-point:hover {
		@apply hover:border-2 hover:border-solid hover:border-red-600;
	}

	.brush {
		stroke: 1px solid gray;
	}
</style>

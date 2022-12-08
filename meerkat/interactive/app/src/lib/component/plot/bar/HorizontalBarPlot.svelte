<script lang="ts">
	import type { Point2D } from '$lib/shared/plot/types';
	import { scaleBand } from 'd3-scale';
	import { Html, LayerCake, Svg } from 'layercake';

	import SvgAxisX from '$lib/shared/plot/layercake/axes/AxisX.svg.svelte';
	import SvgAxisY from '$lib/shared/plot/layercake/axes/AxisY.svg.svelte';
	import SvgBar from '$lib/shared/plot/layercake/bar/Bar.svg.svelte';
	import Label from '$lib/shared/plot/layercake/labels/Label.html.svelte';

	export let data: Array<Point2D>;

	export let padding: number = 10;

	export let xlabel: string = 'x label';
	export let ylabel: string = 'y label';

	let height = 40 * data.length;
</script>

<div class="relative w-full" style:height={`${height}px`}>
	<LayerCake
		ssr={false}
		percentRange={false}
		padding={{ top: padding, right: padding, bottom: padding, left: padding }}
		x="x"
		y="y"
		yScale={scaleBand().paddingInner([0.05])}
		{data}
		xDomain={[-0.4, 0.4]}
		position="absolute"
	>
		<Html>
			<Label axis="x" label={xlabel} />
			<Label axis="y" label={ylabel} />
		</Html>
		<Svg>
			<SvgAxisX />
			<SvgAxisY gridlines={false} />
			<SvgBar on:selection-change />
		</Svg>
	</LayerCake>
</div>

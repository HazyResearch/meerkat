<script lang="ts">
	import type { Point2D } from '$lib/components/plot/types';
	import { scaleBand } from 'd3-scale';
	import { Html, LayerCake, Svg } from 'layercake';

	import SvgAxisX from '$lib/components/plot/layercake/axes/AxisX.svg.svelte';
	import SvgAxisY from '$lib/components/plot/layercake/axes/AxisY.svg.svelte';
	import SvgBar from '$lib/components/plot/layercake/bar/Bar.svg.svelte';
	import Label from '$lib/components/plot/layercake/labels/Label.html.svelte';
	import HtmlAxisY from '$lib/components/plot/layercake/axes/AxisY.html.svelte';
	import FancyHtmlAxisY from './axes/FancyAxisY.html.svelte';
	import FancyLabelY from './labels/FancyLabelY.html.svelte';

	export let data: Array<Point2D>;

	export let padding: number = 10;

	// Width of the "fancy" y ticks in pixels
	export let ywidth: number = 128;

	export let selection: Array<number> = [];

	export let xlabel: string = 'x label';
	export let ylabel: string = 'y label';

	let height = 40 * data.length;
</script>

<div class="relative w-full" style:height={`${height}px`}>
	<LayerCake
		ssr={true}
		percentRange={true}
		padding={{ bottom: padding, left: padding + ywidth }}
		x="x"
		y="y"
		yScale={scaleBand().paddingInner(0.05)}
		{data}
		xDomain={[-0.4, 0.4]}
		position="absolute"
	>
		<Html>
			<!-- <HtmlAxisY /> -->
			<FancyHtmlAxisY width={ywidth} tickMarks={true} />
		</Html>
	</LayerCake>
	<LayerCake
		ssr={false}
		percentRange={false}
		padding={{ bottom: padding, left: padding + ywidth }}
		x="x"
		y="y"
		yScale={scaleBand().paddingInner(0.05)}
		{data}
		xDomain={[-0.4, 0.4]}
		position="absolute"
	>
		<Html>
			<Label axis="x" label={xlabel} />
			<FancyLabelY label={ylabel} offset={0} />
		</Html>
		<Svg>
			<SvgAxisX />
			<!-- <SvgAxisY /> -->
			<SvgBar on:selection-change />
		</Svg>
	</LayerCake>
</div>

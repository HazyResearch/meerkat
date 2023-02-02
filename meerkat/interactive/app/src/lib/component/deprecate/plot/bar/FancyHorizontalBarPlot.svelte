<script lang="ts">
	import { setContext } from 'svelte';
	import type { Point2D } from '$lib/shared/plot/types';
	import { scaleBand } from 'd3-scale';
	import { Html, LayerCake, Svg } from 'layercake';

	import SvgAxisX from '$lib/shared/plot/layercake/axes/AxisX.svg.svelte';
	import SvgAxisY from '$lib/shared/plot/layercake/axes/AxisY.svg.svelte';
	import SvgBar from '$lib/shared/plot/layercake/bar/Bar.svg.svelte';
	import Label from '$lib/shared/plot/layercake/labels/Label.html.svelte';
	import HtmlAxisY from '$lib/shared/plot/layercake/axes/AxisY.html.svelte';
	import FancyHtmlAxisY from './axes/FancyAxisY.html.svelte';
	import FancyLabelY from './labels/FancyLabelY.html.svelte';

	export let data: Array<Point2D>;
	export let metadata: Array<any>;

	export let padding: number = 0;

	// Width of the "fancy" y ticks in pixels
	export let ywidth: number = 256;

	export let selection: Array<number> = [];

	export let xlabel: string = 'x label';
	export let ylabel: string = 'y label';

	export let can_remove: boolean = true;

	const floor = (x: number, decimals: number): number => {
		return Math.floor(x * Math.pow(10, decimals)) / Math.pow(10, decimals);
	};
	const ceil = (x: number, decimals: number): number => {
		return Math.floor(x * Math.pow(10, decimals)) / Math.pow(10, decimals);
	};

	let get_x_domain = () => {
		let min = Math.min(...data.map((d) => d.x));
		let max = Math.max(...data.map((d) => d.x));
		// return [0, max]
		return [Math.min(min, -max) - 0.05, Math.max(max, -min) + 0.05];
		// return [floor(min, 1), ceil(max, 1)];
	}

	// Set a context to allow passing of metadata.
	const metadataContext = setContext("FancyHorizontalBarPlotMetadata", { metadata } );
	const can_remove_context = setContext("can_remove", can_remove)
</script>

<div class="relative h-full w-full z-10">
		<LayerCake
			ssr={true}
			percentRange={true}
			padding={{top: 50, left: padding + ywidth }}
			x="x"
			y="id"
			yScale={scaleBand().paddingInner(0.05)}
			{data}
			xDomain={get_x_domain()}
			position="absolute"
		>
			<Html>
				<!-- <HtmlAxisY /> -->
				<FancyHtmlAxisY width={ywidth} tickMarks={true} on:remove />
			</Html>
		</LayerCake>
		<LayerCake
			ssr={false}
			percentRange={false}
			padding={{top: 50, left: padding + ywidth }}
			x="x"
			y="id"
			yScale={scaleBand().paddingInner(0.05)}
			{data}
			xDomain={get_x_domain()}
			position="absolute"
		>
			<Html>
				<Label axis="x" label={xlabel} />
				<!-- <FancyLabelY label={ylabel} offset={0} /> -->
			</Html>
			<Svg>
				<SvgAxisX />
				<!-- <SvgAxisY /> -->
				<SvgBar on:selection-change />
			</Svg>
		</LayerCake>
</div>

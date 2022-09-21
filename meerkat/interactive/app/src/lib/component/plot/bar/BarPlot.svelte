<script lang="ts">
	import { Html, LayerCake, Svg } from 'layercake';
	import type { Point2D } from '$lib/components/plot/types';
    import { scaleBand } from 'd3-scale';


	import SvgAxisX from './axes/AxisX.svg.svelte';
	import SvgAxisY from './axes/AxisY.svg.svelte';
	import Label from '$lib/components/plot/layercake/labels/Label.html.svelte';
	import SvgBar from './Bar.svg.svelte';

	export let data: Array<Point2D>;

	export let radius = 3;
    export let padding: number = 10;
	export let selection: Array<number> = [];

    export let xlabel: string = 'x label';
    export let ylabel: string = 'y label';
    
	let height = 40 * data.length;



</script>

<div class="w-full" style:height={`${height}px`}>
	<LayerCake
		ssr={false}
		percentRange={false}
		padding={{ top: padding, right: padding, bottom: padding, left: padding }}
		x="x"
		y="y"
        yScale={scaleBand().paddingInner([0.05])}
		{data}
        xDomain={[-0.4, 0.4]}
	>
        <Html>
            <Label axis="x" label={xlabel}/>
            <Label axis="y" label={ylabel}/>
        </Html>
		<Svg>
			<SvgAxisX />
			<SvgAxisY gridlines={false}/>
			<SvgBar {radius} on:selection-change/>
		</Svg>
        
	</LayerCake>
</div>


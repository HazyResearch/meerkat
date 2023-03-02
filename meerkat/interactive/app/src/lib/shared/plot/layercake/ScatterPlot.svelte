<script lang="ts">
	import { Html, LayerCake, Svg } from 'layercake';
	import type { Point2D } from '../types';

	import SvgAxisX from './axes/AxisX.svg.svelte';
	import SvgAxisY from './axes/AxisY.svg.svelte';
	import Label from './labels/Label.html.svelte';
	import SvgScatter from './scatter/Scatter.svg.svelte';

	export let data: Array<Point2D>;

	export let radius = 3;
    export let padding: number = 10;

    export let xlabel: string = 'x label';
    export let ylabel: string = 'y label';
    
    // Height and width of the plot
    export let width: string = '100%';
    export let height: string = '400px';

</script>

<div style:height={height} style:width={width}>
	<LayerCake
		ssr={false}
		percentRange={false}
		padding={{ top: padding, right: padding, bottom: padding, left: padding }}
		x="x"
		y="y"
		{data}
	>
        <Html>
            <Label axis="x" label={xlabel}/>
            <Label axis="y" label={ylabel}/>
        </Html>
		<Svg>
			<SvgAxisX />
			<SvgAxisY />
			<SvgScatter {radius} on:selection-change />
		</Svg>
        
	</LayerCake>
</div>

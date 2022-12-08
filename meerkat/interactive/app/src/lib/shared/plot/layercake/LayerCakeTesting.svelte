<script lang="ts">
	import { Html,LayerCake,Svg } from 'layercake';
	import type { Point2D } from '../types';

	import SvgAxisX from './axes/AxisX.svg.svelte';
	import SvgAxisY from './axes/AxisY.svg.svelte';
	import LabelX from './labels/LabelX.html.svelte';
	import LabelY from './labels/LabelY.html.svelte';
	import SvgScatter from './scatter/Scatter.svg.svelte';

	let data: Array<Point2D> = [
		{
			x: 1979,
			y: 7.19,
			id: 0
		},
		{
			x: 1980,
			y: 7.83,
			id: 1
		},
		{
			x: 1981,
			y: 7.24,
			id: 2
		},
		{
			x: 1982,
			y: 7.44,
			id: 3
		},
		{
			x: 1983,
			y: 7.51,
			id: 4
		},
		{
			x: 1984,
			y: 7.1,
			id: 5
		},
		{
			x: 1985,
			y: 6.91,
			id: 6
		},
		{
			x: 1986,
			y: 7.53,
			id: 0
		},
		{
			x: 1987,
			y: 7.47,
			id: 0
		},
		{
			x: 1988,
			y: 7.48,
			id: 0
		},
		{
			x: 1989,
			y: 7.03,
			id: 0
		},
		{
			x: 1990,
			y: 6.23,
			id: 0
		},
		{
			x: 1991,
			y: 6.54,
			id: 0
		},
		{
			x: 1992,
			y: 7.54,
			id: 0
		},
		{
			x: 1993,
			y: 6.5,
			id: 0
		},
		{
			x: 1994,
			y: 7.18,
			id: 0
		},
		{
			x: 1995,
			y: 6.12,
			id: 0
		},
		{
			x: 1996,
			y: 7.87,
			id: 0
		},
		{
			x: 1997,
			y: 6.73,
			id: 0
		},
		{
			x: 1998,
			y: 6.55,
			id: 0
		},
		{
			x: 1999,
			y: 6.23,
			id: 0
		},
		{
			x: 2000,
			y: 6.31,
			id: 0
		},
		{
			x: 2001,
			y: 6.74,
			id: 0
		},
		{
			x: 2002,
			y: 5.95,
			id: 0
		},
		{
			x: 2003,
			y: 6.13,
			id: 0
		},
		{
			x: 2004,
			y: 6.04,
			id: 0
		},
		{
			x: 2005,
			y: 5.56,
			id: 0
		},
		{
			x: 2006,
			y: 5.91,
			id: 0
		},
		{
			x: 2007,
			y: 4.29,
			id: 0
		},
		{
			x: 2008,
			y: 4.72,
			id: 0
		},
		{
			x: 2009,
			y: 5.38,
			id: 0
		},
		{
			x: 2010,
			y: 4.92,
			id: 0
		},
		{
			x: 2011,
			y: 4.61,
			id: 0
		},
		{
			x: 2012,
			y: 3.62,
			id: 0
		},
		{
			x: 2013,
			y: 5.35,
			id: 0
		},
		{
			x: 2014,
			y: 5.28,
			id: 0
		},
		{
			x: 2015,
			y: 4.63,
			id: 0
		},
		{
			x: 2016,
			y: 4.72,
			id: 0
		}
	];
	data = data.map((d, i) => ({ ...d, id: i }));

	const r = 14;
	const padding = 25;

	let evt;
	let hideTooltip = true;
</script>

<div class="w-full h-80 relative">
	<LayerCake
		ssr={false}
		percentRange={false}
		padding={{ top: 20, right: 20, bottom: 20, left: 20 }}
		x="x"
		y="y"
		{data}
		position="absolute"
	>
		<!-- <Canvas>
			<ScatterCanvas />
		</Canvas> -->
		<Svg zIndex={10}>
			<SvgAxisX />
			<SvgAxisY />
			<SvgScatter />
			<!-- <Brush/> -->
		</Svg>
		<!-- <Html pointerEvents={false}>
			{#if !hideTooltip}
				<Tooltip {evt} let:detail>
					Tooltip
				</Tooltip>
			{/if}
		</Html> -->
	</LayerCake>
	<!-- <LayerCake
		ssr={true}
		percentRange={false}
		padding={{ top: 20, right: 20, bottom: 20, left: 20 }}
		x="x"
		y="y"
		{data}
		position="absolute"
	>
		<Canvas>
			<ScatterCanvas />
		</Canvas>
	</LayerCake> -->
	<LayerCake
		ssr={true}
		percentRange={true}
		padding={{ top: 20, right: 20, bottom: 20, left: 20 }}
		x="x"
		y="y"
		{data}
		position="absolute"
	>
		<Html zIndex={0}>
			<!-- <AxisX />
			<AxisY /> -->
			<LabelX />
			<LabelY />
		</Html>
		<!-- <Html zIndex={0}><Scatter --radius={`${r}px`} --z={200000} /></Html> -->
	</LayerCake>
</div>

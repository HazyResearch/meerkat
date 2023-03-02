<!--
  @component
  Generates an HTML y-axis.
 -->
<script>
	import { getContext } from 'svelte';
	import FancyTick from './FancyTick.svelte';

	const { padding, xRange, yScale } = getContext('LayerCake');

	const { data } = getContext('LayerCake');
	$: metadata = getContext('FancyHorizontalBarPlotMetadata').metadata;
	

	/** @type {Boolean} [gridlines=true] - Extend lines from the ticks into the chart space */
	export let gridlines = true;

	/** @type {Boolean} [tickMarks=false] - Show a vertical mark for each tick. */
	export let tickMarks = false;

	/** @type {Boolean} [baseline=false] – Show a solid line at the bottom. */
	export let baseline = false;

	/** @type {Number|Array|Function} [ticks=4] - If this is a number, it passes that along to the [d3Scale.ticks](https://github.com/d3/d3-scale) function. If this is an array, hardcodes the ticks to those values. If it's a function, passes along the default tick values and expects an array of tick values in return. */
	export let ticks = 4;

	/** @type {Number} [xTick=-4] - How far over to position the text marker. */
	export let xTick = -4;

	/** @type {Number} [yTick=-1] - How far up and down to position the text marker. */
	export let yTick = -1;

	export let width = 40;

	export let close = true;


	$: isBandwidth = typeof $yScale.bandwidth === 'function';

	$: tickVals = Array.isArray(ticks)
		? ticks
		: isBandwidth
		? $yScale.domain()
		: typeof ticks === 'function'
		? ticks($yScale.ticks())
		: $yScale.ticks(ticks);
	
</script>

<div class="axis y-axis">
<!-- <div class="axis y-axis" style="transform:translate(-{$padding.left / 2}px, 0)"> -->
	{#each tickVals as tick, i}
		<div
			class="tick tick-{i}"
			style="
			top:{$yScale(tick) + (isBandwidth ? $yScale.bandwidth() / 2 : 0)}%;
			left:{$xRange[0]}%; 
			height:{$yScale.bandwidth()}%;
			"
		>
			{#if gridlines !== false}
				<div
					class="gridline"
					style="top:0;left:0px;right:-{$padding.right}px;"
				/>
			{/if}
			{#if baseline !== false && i === 0}
				<div
					class="gridline baseline"
					style="top:0;left:0px;right:-{$padding.right}px;"
				/>
			{/if}
			{#if tickMarks === true}
				<div
					class="tick-mark"
					style="top:0;left:-6px;width:6px;"
				/>
			{/if}
			<div
				class="text h-full z-20"
				style="
			width:{width}px;
			transform: translate(
				{isBandwidth ? '-110%' : 0}, 
				{isBandwidth ? -50 - Math.floor($yScale.bandwidth() / -2) : '-100'}%);
			"
			>	
				<!-- 
					Make a FancyTick, which represents any component that should be shown on the left 
					side of the plot. This has a fixed width in pixels. 

					TODO (arjundd): Make a default value of count so that it doesn't display.
				-->
				<FancyTick width="{width}px" name={$data[i].y} id={$data[i].id} size={metadata[i][0] || 0} on:remove />
			</div>
		</div>
	{/each}
</div>

<style>
	.axis,
	.tick,
	.tick-mark,
	.gridline,
	.baseline,
	.text {
		position: absolute;
	}
	.axis {
		width: 100%;
		height: 100%;
	}
	.tick {
		width: 100%;
	}

	.gridline {
		border-top: 1px dashed #aaa;
	}
	.tick-mark {
		border-top: 1px solid #aaa;
	}

	.baseline.gridline {
		border-top-style: solid;
	}

	.tick .text {
		color: #666;
	}
</style>

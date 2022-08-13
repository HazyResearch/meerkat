<script lang="ts">
	import { brush, brushX, brushY } from 'd3-brush';
	import { select } from 'd3-selection';
	import { getContext } from 'svelte';

	const { width, height } = getContext('LayerCake');

	export let brushed: (event: any) => void;
	export let brush_x: boolean = true;
	export let brush_y: boolean = true;
	export let brush_element: any = undefined;
    export let fire_on: string = 'start brush end';

	// Setup the brush using d3
	// .extent defines the viewport of the brush
	// .on binds a function to the brush events
	// .filter overrides the default .filter that prevents the brush
	// from activating when the CTRL key is pressed
	$: {
		let brush_fn = brush_x && brush_y ? brush : brush_x ? brushX : brushY;

		let _brush = brush_fn()
			.extent([
				[-1, -1],
				[$width + 1, $height + 1]
			])
			.on(fire_on, brushed)
			.filter((event) => !event.button);

		if (brush_element) {
			select(brush_element).call(_brush);
		}
	}
</script>

<g class="brush" bind:this={brush_element} />

<!-- { filename: './shared/CanvasLayer.svelte' } -->
<script>
	import { scaleCanvas } from 'layercake';
	import { getContext } from 'svelte';

	const { data, xGet, yGet, yScale, width, height } = getContext('LayerCake');
	const { ctx } = getContext('canvas');

	$: {
		if ($ctx) {
			/* --------------------------------------------
			 * If you were to have multiple canvas layers
			 * maybe for some artistic layering purposes
			 * put these reset functions in the first layer, not each one
			 * since they should only run once per update
			 */
			scaleCanvas($ctx, $width, $height);
			$ctx.clearRect(0, 0, $width, $height);

			/* --------------------------------------------
			 * Draw the scatterplot
			 */
			$data.forEach((d) => {
				$ctx.beginPath();
				$ctx.arc($xGet(d), $yGet(d), 5, 0, 2 * Math.PI, false);
				$ctx.fillStyle = '#f0c';
				$ctx.fill();
			});
		}
	}
</script>
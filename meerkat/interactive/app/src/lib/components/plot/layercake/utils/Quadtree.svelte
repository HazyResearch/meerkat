<svelte:options accessors={true} />

<script lang="ts">
	import { quadtree, type Quadtree } from 'd3-quadtree';
	import { getContext } from 'svelte';
	import type { Point2D } from '../../types';

	const { data, xGet, yGet, width, height } = getContext('LayerCake');

	export let tree: Quadtree<Point2D>;

	$: tree = quadtree()
		.extent([
			[-1, -1],
			[$width + 1, $height + 1]
		])
		.x($xGet)
		.y($yGet)
		.addAll($data);

	// Find the nodes within a specified rectangle.
	// Taken from https://observablehq.com/@d3/quadtree-brush
	export function search([[x0, y0], [x3, y3]]: any) {
		let selected_points = new Set();
		let scanned_points = new Set();
		tree.visit((node: any, x1: any, y1: any, x2: any, y2: any) => {
			if (!node.length) {
				do {
					const x = $xGet(node.data);
					const y = $yGet(node.data);
					const id = node.data.id;
					// d.scanned = true;
					// d.selected = x >= x0 && x < x3 && y >= y0 && y < y3;
					scanned_points.add(id);
					if (x >= x0 && x <= x3 && y >= y0 && y <= y3) {
						selected_points.add(id);
					}
				} while ((node = node.next));
			}
			return x1 >= x3 || y1 >= y3 || x2 < x0 || y2 < y0;
		});
		return [selected_points, scanned_points];
	}
</script>

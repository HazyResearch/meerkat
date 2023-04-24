<script lang="ts">
	import type { Box } from '$lib/utils/annotations';
	import {
		convertImageCoordinatesToClickCoordinates,
		convertClickCoordinatesToImageCoordinates
	} from '$lib/utils/coordinates';

	export let boxes: Array<Box>;
	export let categories: object;
	export let image: HTMLImageElement | null;
	export let isActive: boolean = true;
	export let canvas: HTMLCanvasElement;
	export let selectedCategory: string | null = null;


	function toHex(rgb: Array<number>) {
		const r = rgb[0];
		const g = rgb[1];
		const b = rgb[2];

		return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
	}

	let selected: Array<object> = [];
	$: display = image ? convertToDisplay(boxes, image) : [];
	$: context = canvas ? canvas.getContext('2d') : null;
	$: selectedCategoryColor = selectedCategory ? toHex(categories[selectedCategory]) : '#000000';

	let startX: number;
	let startY: number;
	let isDrawing: boolean;

	function manageEventListeners(canvas, isActive: boolean) {
		if (!canvas) return;

		if (isActive) {
			canvas.addEventListener('mousedown', startDrawing);
			canvas.addEventListener('mousemove', draw);
			canvas.addEventListener('mouseup', endDrawing);
		} else {
			canvas.removeEventListener('mousedown', startDrawing);
			canvas.removeEventListener('mousemove', draw);
			canvas.removeEventListener('mouseup', endDrawing);
		}
	}

	function convertToDisplay(boxes: Array<Box>, _image: HTMLImageElement) {
		const out = boxes.map((box) => {
			let start = convertImageCoordinatesToClickCoordinates({ x: box.x, y: box.y }, _image);
			let end = convertImageCoordinatesToClickCoordinates(
				{ x: box.x + box.width, y: box.y + box.height },
				_image
			);
			return {
				x: start.x,
				y: start.y,
				width: end.x - start.x,
				height: end.y - start.y,
				box: box
			};
		});
		return out;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.code === 'Backspace' || event.code === 'Delete') {
			boxes = boxes.filter((box) => !selected.includes(box));
			selected = [];
		}

		if (event.code === 'KeyA' && event.ctrlKey) {
			selected = boxes;
		}
	}

	function startDrawing(event) {
		isDrawing = true;
		startX = event.offsetX;
		startY = event.offsetY;

		context.strokeStyle = selectedCategoryColor;
		context.lineWidth = 1;
	}

	function draw(event) {
		// Draw the bounding box
		if (isDrawing) {
			context.clearRect(0, 0, canvas.width, canvas.height);
			// Draw the bounding box
			const width = event.offsetX - startX;
			const height = event.offsetY - startY;
			context.strokeRect(startX, startY, width, height);
		}
	}

	function endDrawing(event) {
		isDrawing = false;
		context.clearRect(0, 0, canvas.width, canvas.height);

		// Add the bounding box to the list of boxes.
		const start = convertClickCoordinatesToImageCoordinates({ x: startX, y: startY }, image);
		const end = convertClickCoordinatesToImageCoordinates(
			{ x: event.offsetX, y: event.offsetY },
			image
		);

		// Don't add a box if the user didn't draw anything.
		if (start.x === end.x || start.y === end.y) return;

		boxes = [
			...boxes,
			{
				x: Math.min(start.x, end.x),
				y: Math.min(start.y, end.y),
				width: Math.abs(end.x - start.x),
				height: Math.abs(end.y - start.y),
				category: selectedCategory || ''
			}
		];

	}

	function getColor(displayBox, categories) {

		const boxColor = categories[displayBox.box.category]
			? toHex(categories[displayBox.box.category])
			: 'black';
        
		return selected.includes(displayBox) ? 'blue' : boxColor;
	}

    function handleSelect(box) {
        if (selected.includes(box)) {
            selected = selected.filter((x) => x !== box);
        } else {
            selected = [...selected, box];
        }
    }

	$: manageEventListeners(canvas, isActive);
</script>

<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
<div on:keydown={handleKeydown} tabindex="0">
	{#each display as box}
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<div
			class="box"
			style="left: {box.x}px; top: {box.y}px; width: {box.width}px; height: {box.height}px; border: 3px solid {selected.includes(box.box) ? '#0000FF' : getColor(
				box,
				categories,
			)};"
            on:click={(e) => {
                handleSelect(box.box);
                e.stopPropagation();
            }}
		/>
	{/each}
</div>

<style>
	.box {
		position: absolute;
	}

</style>

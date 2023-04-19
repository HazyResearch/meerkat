<!-- Image Annotator Component

	This component was inspired by Gradio's AnnotatedImage.
 -->
<script lang="ts">
	import Toolbar from '$lib/shared/common/Toolbar.svelte';
	import { KeyCode } from 'monaco-editor';
	import { onMount } from 'svelte';
	import { Palette } from 'svelte-bootstrap-icons';
	import { uniqueId } from 'underscore';

	export let data: string;
	export let categories: object;
	export let segmentations: Array<object>;
	export let opacity: number = 0.85;
	export let toolbar: Array<string> = ['segmentation', 'select'];
	export let points: Array<{ x: number; y: number; color?: number | string }> = [];
	export let pointSize: number = 10;
	export let selectedCategory: string | null = null;

	const baseImageId = uniqueId('image-');
	let baseImageElement: HTMLImageElement | null = null;

	$: labels = [...new Set(segmentations.map((arr) => arr[1]))];

	// Coordinates for the points are relative to the image's bounding box.
	let imageRectHeight = baseImageElement?.height;
	let imageRectWidth = baseImageElement?.width;
	$: displayPoints = baseImageElement ? convertToDisplayPoints(points, baseImageElement?.getBoundingClientRect()) : [];
	let selectedPoints: Array<object> = [];

	let activeCategories: Array<string> = [];
	let temporaryActiveCategory: string | null = null;
	function handleCategoryMouseover(label: string) {
		temporaryActiveCategory = label;
	}
	function handleCategoryMouseout() {
		temporaryActiveCategory = null;
	}
	function handleCategoryClicked(label: string) {
		if (activeCategories.includes(label)) {
			activeCategories = activeCategories.filter((category) => category !== label);
		} else {
			activeCategories = [...activeCategories, label];
		}
	}

	function getHexColor(color: Array<number>, isActive: boolean | null) {
		const r = color[0];
		const g = color[1];
		const b = color[2];

		const base = '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
		if (isActive === null) {
			return base;
		}

		let opacity = isActive ? 'a0' : '60';
		return base + opacity;
	}

	function hex2rgba(hex: string, opacity: number) {
		let r = parseInt(hex.slice(1, 3), 16);
		let g = parseInt(hex.slice(3, 5), 16);
		let b = parseInt(hex.slice(5, 7), 16);

		return [r, g, b, 255];
	}

	// Color picker logic
	function handleColorChange(label: string, hexColor: string) {
		categories[label] = hex2rgba(hexColor);
	}

	function convertToDisplayPoints(points: Array<{ x: number; y: number }>, imageRect: DOMRect) {
		const out = points.map((point) => {
			let clickCoordinates = convertImageCoordinatesToClickCoordinates(point.x, point.y, imageRect)
			clickCoordinates["point"] = point;
			return clickCoordinates;
		});
		console.log(out);
		return out;
	}

	function convertImageCoordinatesToClickCoordinates(
		xImage: number,
		yImage: number,
		imageRect: DOMRect
	) {
		// Larger values means the image is scaled down more.
		// The larger ratio indicates the biggest resize that was
		// applied to the image.
		const heightRatio = baseImageElement.naturalHeight / imageRect.height;
		const widthRatio = baseImageElement.naturalWidth / imageRect.width;
		const ratio = Math.max(heightRatio, widthRatio);

		// The shape of the displayed image.
		// We assume the image is displayed with `contain` bounds.
		// This means the image will be scaled (preserving aspect ratio) to fit in the container.
		const imageHeight = baseImageElement.naturalHeight / ratio;
		const imageWidth = baseImageElement.naturalWidth / ratio;
		// padding should never be less than 0.
		const padTop = (imageRect.height - imageHeight) / 2;
		const padLeft = (imageRect.width - imageWidth) / 2;

		// The coordinates of the click relative to original image shape.
		const x = xImage / ratio + padLeft;
		const y = yImage / ratio + padTop;
		return {"x": x, "y": y}
	}

	function convertClickCoordinatesToImageCoordinates(
		x: number,
		y: number,
		image: HTMLImageElement
	) {
		const imageRect = image.getBoundingClientRect();

		// Larger values means the image is scaled down more.
		// The larger ratio indicates the biggest resize that was
		// applied to the image.
		const heightRatio = image.naturalHeight / imageRect.height;
		const widthRatio = image.naturalWidth / imageRect.width;
		const ratio = Math.max(heightRatio, widthRatio);

		// The shape of the displayed image.
		// We assume the image is displayed with `contain` bounds.
		// This means the image will be scaled (preserving aspect ratio) to fit in the container.
		const imageHeight = image.naturalHeight / ratio;
		const imageWidth = image.naturalWidth / ratio;
		// padding should never be less than 0.
		const padTop = (imageRect.height - imageHeight) / 2;
		const padLeft = (imageRect.width - imageWidth) / 2;

		// The coordinates of the click relative to original image shape.
		const xImage = (x - padLeft) * ratio;
		const yImage = (y - padTop) * ratio;

		return {"x": xImage, "y": yImage}
	}

	// Handle selecting a point on the image.
	function handleSelect(event: PointerEvent) {
		const imageCoordinates = convertClickCoordinatesToImageCoordinates(
			event.offsetX,
			event.offsetY,
			event.target
		);
		console.log(points)
		points = [...points, imageCoordinates];
	}

	function handleSelectPoint(point) {
		if (selectedPoints.includes(point)) {
			selectedPoints = selectedPoints.filter((p) => p !== point);
		} else {
			selectedPoints = [...selectedPoints, point];
		}
	}

	function handleKeydownPoint(event: KeyboardEvent) {
		if (event.code === "Backspace" || event.code === "Delete") {
			points = points.filter((point) => !selectedPoints.includes(point));
			selectedPoints = [];
		}

		if (event.code === "KeyA" && event.ctrlKey) {
			selectedPoints = points;
		}
	}

	onMount(() => {
		console.log('mounting', baseImageId, document.getElementById(baseImageId));
		baseImageElement = document.getElementById(baseImageId);
		imageRectHeight = baseImageElement.height;
		imageRectWidth = baseImageElement.width;

		// be smarter about this.
		points = [...points];

		const observer = new ResizeObserver((entries) => {
			for (let entry of entries) {
				if (entry.target === baseImageElement) {
					imageRectHeight = baseImageElement.height;
					imageRectWidth = baseImageElement.width;
					points = [...points];
					console.log('image dimensions', imageRectHeight, imageRectWidth);
				}
			}
		});
		observer.observe(baseImageElement);
	});

</script>

<div class="flex flex-col gap-y-2 bg-white w-full h-full">
	<!-- Display -->
	<div class="image-container">
		<!-- svelte-ignore a11y-missing-attribute -->
		<img id={baseImageId} src={data} on:click={handleSelect} />

		<!-- Segmentations -->
		{#each segmentations ? segmentations : [] as [seg, label]}
			<!-- svelte-ignore a11y-missing-attribute -->
			<img
				on:click={handleSelect}
				class="mask cursor-pointer"
				class:visible={activeCategories.includes(label) || label == temporaryActiveCategory}
				class:invisible={!activeCategories.includes(label) &&
					temporaryActiveCategory != null &&
					temporaryActiveCategory != label}
				src={seg}
			/>
		{/each}

		<!-- Points -->
		<div on:keydown={handleKeydownPoint}  tabindex="0">
		{#each displayPoints as point}
			<div
				style="position: absolute; left: {point.x - pointSize / 2}px; top: {point.y -
					pointSize / 2}px;"
				on:click={handleSelectPoint(point.point)}
			>
				<div
					style="width: {pointSize}px; height: {pointSize}px; background-color: red; border-radius: 50%; border: 2px solid {selectedPoints.includes(point.point) ? 'blue' : 'black'};"
				/>
			</div>
		{/each}
	</div>
	</div>

	<!-- Add toolbar for opacity, etc. -->
	<!-- <div>
		<Toolbar isToolbarActive={true} classes="px-3" align="bottom" pin={true}>
			<button class="" on:click={() => {}}>
				<Palette width={24} height={24} fill="black" />
			</button>
		</Toolbar>
	</div> -->

	<!-- Legend -->
	<div class="flex flex-row flex-wrap content-center justify-center gap-2 m-2">
		{#each labels ? labels : [] as label}
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<!-- TODO: color the button based on key -->
			<div
				class="w-fit py-1 px-2 flex items-center text-slate-600 rounded-md cursor-pointer bg-[{getHexColor(
					categories[label],
					activeCategories.includes(label) || label == temporaryActiveCategory
				)}] legend-item"
				on:mouseover={() => handleCategoryMouseover(label)}
				on:focus={() => handleCategoryMouseover(label)}
				on:mouseout={() => handleCategoryMouseout()}
				on:blur={() => handleCategoryMouseout()}
				on:click={() => handleCategoryClicked(label)}
			>
				{label}
				<input
					type="color"
					on:change={(e) => handleColorChange(label, e.target.value)}
					id="color-picker"
					value={getHexColor(categories[label], null)}
				/>
			</div>
		{/each}
	</div>
</div>

<style>
	.container {
		display: flex;
		position: relative;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		width: var(--size-full);
		height: var(--size-full);
	}
	.image-container {
		position: relative;
		top: 0;
		left: 0;
		flex-grow: 1;
		width: 100%;
		overflow: hidden;
		background-color: #000000;
	}
	.image-container img {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	.image-container:hover .mask {
		opacity: 0.3;
	}

	.mask {
		opacity: 0;
		transition: all 0.2s ease-in-out;
	}
	.mask.visible {
		opacity: 0.85;
	}
	.mask.invisible {
		opacity: 0;
	}

	#color-picker {
		-webkit-appearance: none;
		-moz-appearance: none;
		appearance: none;
		background-color: transparent;
		border: none;
		cursor: pointer;
		width: 20px;
		height: 20px;
	}
	#color-picker::-webkit-color-swatch {
		border-radius: 50%;
		border: none;
		border: 2px solid #000000;
	}
	#color-picker::-moz-color-swatch {
		border-radius: 50%;
		border: none;
		border: 2px solid #000000;
	}
</style>

<!-- Image Annotator Component

	This component was inspired by Gradio's AnnotatedImage.
 -->
<script lang="ts">
	import Toolbar from '$lib/shared/common/Toolbar.svelte';
	import type { Endpoint } from '$lib/utils/types';
	import { createEventDispatcher, onMount } from 'svelte';
	import {
		Eye,
		EyeSlash,
		Palette,
		Plus,
		BoundingBoxCircles,
		Cursor,
		HandIndex
	} from 'svelte-bootstrap-icons';
	import { uniqueId } from 'underscore';
	import { dispatch } from '$lib/utils/api';
	import {
		convertImageCoordinatesToClickCoordinates,
		convertClickCoordinatesToImageCoordinates
	} from '$lib/utils/coordinates';
	import PointLayer from './PointLayer.svelte';
	import BoxLayer from './BoxLayer.svelte';
	import type { Box } from '$lib/utils/annotations';
	import Button from '../button/Button.svelte';

	export let data: string;
	export let categories: object;
	export let segmentations: Array<object>;
	export let boxes: Array<Box> = [];
	export let opacity: number = 0.85;
	export let toolbar: Array<string> = ['segmentation', 'point', 'box'];
	export let points: Array<{ x: number; y: number; color?: number | string }> = [];
	export let pointSize: number = 10;
	export let selectedCategory: string | null = '';
	export let selectedTool: string = '';

	export let onAddCategory: Endpoint = null;
	export let onAddBox: Endpoint = null;
	export let onAddPoint: Endpoint = null;

	const baseImageId = uniqueId('image-');
	const canvasId = uniqueId('canvas-');

	$: console.log("all the boxes", boxes)

	let baseImageElement: HTMLImageElement | null = null;
	let canvasElement: HTMLCanvasElement | null = null;
	$: showCanvas = selectedTool === 'box' || selectedTool === 'point';

	$: labels =
		Object.keys(categories).length > 10
			? [...new Set(segmentations.map((arr) => arr[1]))]
			: [...Object.keys(categories)];

	// Coordinates for the points are relative to the image's bounding box.
	// TODO: delete this
	let imageRectHeight = baseImageElement?.height;
	let imageRectWidth = baseImageElement?.width;

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

	let showNewCategoryTextbox: boolean = false;
	function addNewCategory(e) {
		const text = e.target.value;
		if (text === null || text == undefined || text === '') {
			return;
		}
		const promise = dispatch(onAddCategory.endpointId, {
			detail: { category: text }
		});
		promise.then(() => {
			selectedCategory = text;
			activeCategories = [...activeCategories, text];
		});
		e.target.value = '';
		showNewCategoryTextbox = false;
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

	function handleSelectedCategory(label: string) {
		if (selectedCategory === label) {
			selectedCategory = ''; // we should be able to set this to null
		} else {
			selectedCategory = label;
		}
	}

	function selectTool(tool: string) {
		if (selectedTool === tool) {
			selectedTool = '';
		} else {
			selectedTool = tool;
		}
	}

	onMount(() => {
		baseImageElement = document.getElementById(baseImageId);
		canvasElement = document.getElementById(canvasId);
		imageRectHeight = baseImageElement.height;
		imageRectWidth = baseImageElement.width;
		canvasElement.height = imageRectHeight;
		canvasElement.width = imageRectWidth;

		// be smarter about this.
		// points = [...points];

		const observer = new ResizeObserver((entries) => {
			for (let entry of entries) {
				if (entry.target === baseImageElement) {
					imageRectHeight = baseImageElement.height;
					imageRectWidth = baseImageElement.width;
					baseImageElement = entry.target;

					// Setting canvas properties.
					canvasElement.height = imageRectHeight;
					canvasElement.width = imageRectWidth;
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
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<img id={baseImageId} src={data} />

		<!-- Segmentations -->
		{#each segmentations ? segmentations : [] as [seg, label]}
			<!-- svelte-ignore a11y-missing-attribute -->
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<img
				class="mask cursor-pointer"
				class:visible={activeCategories.includes(label) || label == temporaryActiveCategory}
				class:invisible={!activeCategories.includes(label) &&
					temporaryActiveCategory != null &&
					temporaryActiveCategory != label}
				src={seg}
			/>
		{/each}

		<!-- Box -->
		<BoxLayer
			bind:boxes
			bind:categories
			bind:image={baseImageElement}
			bind:canvas={canvasElement}
			bind:selectedCategory
			isActive={selectedTool == 'box'}
		/>

		<!-- Points -->
		<PointLayer
			bind:points
			bind:image={baseImageElement}
			bind:pointSize
			bind:canvas={canvasElement}
			isActive={selectedTool == 'point'}
		/>

		<canvas
			id={canvasId}
			bind:this={canvasElement}
			style={showCanvas ? '' : 'display: none;'}
			width="200"
			height="200"
		/>
	</div>

	<!-- Add toolbar for opacity, etc. -->
	<div class="flex self-center gap-x-2 mx-2 bg-slate-100 w-fit p-1 rounded-md">
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<div
			class="flex p-1 hover:bg-[#BDE1F4] rounded-md {selectedTool == 'select'
				? 'bg-[#BDE1F4]'
				: ''}"
			on:click={() => {
				selectTool('select');
			}}
		>
			<Cursor width={24} height={24} fill="black" />
		</div>

		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<div
			class="flex p-1 hover:bg-[#BDE1F4] rounded-md {selectedTool == 'point' ? 'bg-[#BDE1F4]' : ''}"
			on:click={() => {
				selectTool('point');
			}}
		>
			<HandIndex width={24} height={24} fill="black" />
		</div>

		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<div
			class="flex p-1 hover:bg-[#BDE1F4] rounded-md {selectedTool == 'box' ? 'bg-[#BDE1F4]' : ''}"
			on:click={() => {
				selectTool('box');
			}}
		>
			<BoundingBoxCircles width={24} height={24} fill="black" />
		</div>
	</div>

	<!-- Legend -->
	<div class="flex flex-row flex-wrap content-center justify-center gap-2 m-2">
		{#each labels ? labels : [] as label}
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<!-- TODO: color the button based on key -->
			<div
				class="w-fit py-1 px-2 flex items-center text-slate-600 rounded-md gap-x-2 cursor-pointer bg-[{getHexColor(
					categories[label],
					selectedCategory === label
				)}] legend-item"
				on:mouseover={() => handleCategoryMouseover(label)}
				on:focus={() => handleCategoryMouseover(label)}
				on:mouseout={() => handleCategoryMouseout()}
				on:blur={() => handleCategoryMouseout()}
			>
				<div on:click={() => handleCategoryClicked(label)}>
					{#if activeCategories.includes(label)}
						<Eye fill="black" />
					{:else}
						<EyeSlash fill="black" />
					{/if}
				</div>
				<div on:click={() => handleSelectedCategory(label)}>
					{label}
				</div>
				<input
					type="color"
					on:change={(e) => handleColorChange(label, e.target.value)}
					id="color-picker"
					value={getHexColor(categories[label], null)}
				/>
			</div>
		{/each}

		<!-- Add new label button -->
		<div class="w-fit py-1 px-2 flex items-center text-slate-600 rounded-md gap-x-2 cursor-pointer">
			<button
				on:click={() => {
					showNewCategoryTextbox = true;
				}}
			>
				<div>
					<Plus fill="black" />
				</div>
			</button>
			{#if showNewCategoryTextbox}
				<input
					type="text"
					on:keydown={(e) => {
						if (e.code === 'Enter') {
							addNewCategory(e);
						}
					}}
					on:blur={(e) => addNewCategory(e)}
					placeholder="New category"
				/>
			{/if}
		</div>
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

	.image-container canvas {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
	}

	/* .image-container:hover .mask {
		opacity: 0.3;
	} */

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

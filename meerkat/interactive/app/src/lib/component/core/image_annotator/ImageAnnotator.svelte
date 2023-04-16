<!-- Image Annotator Component

	This component was inspired by Gradio's AnnotatedImage.
 -->
<script lang="ts">
	import Toolbar from '$lib/shared/common/Toolbar.svelte';
	import { Palette } from 'svelte-bootstrap-icons';

	export let data: string;
	export let categories: object;
	export let segmentations: Array<object>;
	export let opacity: number = 0.85;
	export let toolbar: Array<string> = ["segmentation", "select"];

	$: labels = [...new Set(segmentations.map((arr) => arr[1]))];

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
		console.log(label, hexColor);
		categories[label] = hex2rgba(hexColor);
	}
</script>

<div class="flex flex-col gap-y-2 bg-white w-full h-full">
	<!-- Display -->
	<div class="image-container">
		<!-- svelte-ignore a11y-missing-attribute -->
		<img src={data} />
		{#each segmentations ? segmentations : [] as [seg, label]}
			<!-- svelte-ignore a11y-missing-attribute -->
			<img
				class="mask"
				class:visible={activeCategories.includes(label) || label == temporaryActiveCategory}
				class:invisible={!activeCategories.includes(label) &&
					temporaryActiveCategory != null &&
					temporaryActiveCategory != label}
				src={seg}
			/>
		{/each}
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

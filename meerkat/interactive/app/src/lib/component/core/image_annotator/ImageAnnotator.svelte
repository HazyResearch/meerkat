<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Button from '../button/Button.svelte';

	export let data: string;
	export let categories: object;
	export let segmentations: Array<object>;
	export let opacity: number = 0.85;

	$: labels = [...new Set(segmentations.map(arr => arr[1]))]

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

	function getHexColor(color: Array<number>, isActive: boolean) {
		let r = color[0];
		let g = color[1];
		let b = color[2];

		let opacity = isActive ? 'a0' : '60';
		return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1) + opacity;
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
				class:invisible={!(activeCategories.includes(label) || label == temporaryActiveCategory)}
				src={seg}
			/>
		{/each}
	</div>

	<!-- Legend -->
	<div class="flex flex-row flex-wrap content-center justify-center gap-2 m-2">
		{#each labels ? labels : [] as label}
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<!-- TODO: color the button based on key -->
			<div
				class="w-fit py-1 px-2 flex items-center text-slate-600 rounded-md cursor-pointer bg-[{getHexColor(
					categories[label],
					activeCategories.includes(label) || label == temporaryActiveCategory,
				)}] legend-item"
				on:mouseover={() => handleCategoryMouseover(label)}
				on:focus={() => handleCategoryMouseover(label)}
				on:mouseout={() => handleCategoryMouseout()}
				on:blur={() => handleCategoryMouseout()}
				on:click={() => handleCategoryClicked(label)}
			>
				{label}
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

	.mask {
		opacity: 0.85;
		transition: all 0.2s ease-in-out;
	}
	.mask.visible {
		opacity: 0.85;
	}
	.mask.invisible {
		opacity: 0;
	}
</style>

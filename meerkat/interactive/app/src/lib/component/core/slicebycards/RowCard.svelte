<script lang="ts">
	import { createTippy } from 'svelte-tippy';
	import { followCursor } from 'tippy.js';

	import Cell from '$lib/shared/cell/Cell.svelte';
	import type { CellInterface } from '$lib/utils/types';
	import type { SvelteComponent } from 'svelte';
	import { openModal } from 'svelte-modals';

	// ID for this component
	export let id: string;

	// Main cell
	export let main: CellInterface;

	// Content cells
	export let tags: Array<CellInterface>;

	// trim long tags – I tried to do this with CSS, which would've been much
	// cleaner (and better since it would've adjusted to the size of the image)
	// but I was running into this weird margin issue that messed up the spacing
	$: tags = tags.map((tag) => {
		if (tag.data.length > 10) {
			tag.data = tag.data.substring(0, 10) + '...';
		}
		return tag;
	});

	// Layout
	export let layout: string;

	// Give the card the `flex-grow` Tailwind class to horizontally
	// fill out space in the (containing) flex container.
	export let card_flex_grow: boolean = false;

	// Tooltip setup
	export let main_tooltip: boolean = false;
	export let tag_tooltip: boolean = true;
	let main_tippy = (node: HTMLElement, parameters: any = null) => {};
	let tag_tippy = (node: HTMLElement, parameters: any = null) => {};

	if (main_tooltip) {
		main_tippy = createTippy({
			placement: 'auto',
			allowHTML: true,
			theme: 'main-tooltip',
			followCursor: true,
			plugins: [followCursor],
			duration: [0, 0],
			maxWidth: '95vw'
			// interactive: true
		});
	}
	if (tag_tooltip) {
		tag_tippy = createTippy({
			allowHTML: true,
			theme: 'tag-tooltip',
			followCursor: true,
			plugins: [followCursor],
			duration: [0, 0],
			maxWidth: '95vw'
		});
	}

	// Modal setup
	export let main_modal: boolean = false;
</script>

<div class="h-full">
	<!-- Main (main) element -->
	<div class="h-4/5 w-max hover:opacity-50">
		<!-- Main item -->
		<Cell {...main} />

		<!-- Main tooltip -->
		{#if main_tooltip}
			<div id="{id}-main-tooltip" class="hidden">
				<slot name="main-tooltip">
					<Cell {...main} />
				</slot>
			</div>
		{/if}
	</div>

	<!-- Content -->
	<div class="h-max w-max flex flex-wrap items-start">
		{#each tags as tag, j}
			<div
				class="mx-1 my-1 px-2 py-0.5 rounded-full text-center text-slate-800 text-xs font-mono bg-violet-200 hover:bg-violet-600 hover:text-slate-200"
				use:tag_tippy={{
					tag: document.getElementById(`${id}-tag-tooltip-${j}`)?.innerHTML
				}}
			>
				<div class="">
					<Cell {...tag} />
				</div>
			</div>
		{/each}
	</div>
</div>

<style>


	/* CSS for the tooltips */
	:global(.tippy-box[data-theme='main-tooltip']) {
		@apply py-1 px-1 text-xs font-mono rounded-lg shadow-sm;
		@apply text-white bg-violet-500;
	}

	:global(.tippy-box[data-theme='tag-tooltip']) {
		@apply py-4 px-4 text-base font-mono rounded-lg shadow-sm;
		@apply text-white bg-violet-900;
	}
</style>

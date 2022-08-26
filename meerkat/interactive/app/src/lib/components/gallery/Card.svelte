<script lang="ts">
	import { createTippy } from 'svelte-tippy';
	import { followCursor } from 'tippy.js';

	import Cell,{ type CellInterface } from '$lib/components/cell/Cell.svelte';
	import type { SvelteComponent } from 'svelte';
	import { openModal } from 'svelte-modals';
	
	// ID for this component
	export let id: string;

	// Pivot cell
	export let pivot: CellInterface;

	// Content cells
	export let content: Array<CellInterface>;

	// Layout
	export let layout: string;

	// Give the card the `flex-grow` Tailwind class to horizontally
	// fill out space in the (containing) flex container.
	export let card_flex_grow: boolean = false;

	// Tooltip setup
	export let pivot_tooltip: boolean = false;
	export let content_tooltip: boolean = true;
	let pivot_tippy = (node: HTMLElement, parameters: any = null) => {};
	let content_tippy = (node: HTMLElement, parameters: any = null) => {};

	if (pivot_tooltip) {
		pivot_tippy = createTippy({
			placement: 'auto',
			allowHTML: true,
			theme: 'pivot-tooltip',
			followCursor: true,
			plugins: [followCursor],
			duration: [0, 0],
			maxWidth: '95vw',
			// interactive: true
		});
	}
	if (content_tooltip) {
		content_tippy = createTippy({
			allowHTML: true,
			theme: 'content-tooltip',
			followCursor: true,
			plugins: [followCursor],
			duration: [0, 0],
			maxWidth: '95vw'
		});
	}

	// Modal setup
	export let pivot_modal: boolean = true;
	export let pivot_modal_component: SvelteComponent;
	export let pivot_modal_component_props: Object;

	// Additional styling props
	export let blur = false;
</script>


<div
	class="mx-2 my-4"
	class:blur-sm={blur}
	class:flex-grow={card_flex_grow}
	class:card-masonry={layout === 'masonry'}
	class:card-gimages={layout === 'gimages'}
>
	<!-- Pivot (main) element -->
	<div
		class="pivot"
		on:click={pivot_modal
			? () => openModal(pivot_modal_component, { is_open: true, ...pivot_modal_component_props })
			: null} 
		use:pivot_tippy={pivot_tooltip
			? { content: document.getElementById(`${id}-pivot-tooltip`)?.innerHTML }
			: null}
	>
		<!-- Pivot item -->
		<Cell {...pivot} />

		<!-- Pivot tooltip -->
		{#if pivot_tooltip}
			<div id="{id}-pivot-tooltip" class="hidden">
				<slot name="pivot-tooltip">
					<Cell {...pivot} />
				</slot>
			</div>
		{/if}
	</div>

	<!-- Content -->
	<div class="content">
		{#each content as subcontent, j}
			<div
				class="mx-1 my-1 px-2 py-0.5 rounded-full text-center text-slate-800 text-xs font-mono bg-violet-200 hover:bg-violet-600 hover:text-slate-200"
				use:content_tippy={{
					content: document.getElementById(`${id}-content-tooltip-${j}`)?.innerHTML
				}}
			>
				<Cell {...subcontent} />
				{#if content_tooltip}
					<div id="{id}-content-tooltip-{j}" class="hidden">
						<Cell {...subcontent} />
					</div>
				{/if}
			</div>
		{/each}
	</div>

</div>

<style>
	.card {
		min-width: var(--card-width, '');
		@apply m-2 border-2 border-solid rounded-lg shadow-md;
		@apply dark:bg-gray-700 dark:border-gray-600;
	}

	.card-masonry {
		/* Solution 1: multiple columns in a masonry layout */
		@apply break-inside-avoid h-auto;
	}

	.card-gimages {
		/* Solution 2: flex containers in the Google Images style */
		/* Make the card a flex-col so the pivot element can be centered horizontally */
		@apply flex flex-col;
	}

	.pivot {
		@apply self-center;
		@apply border-2 border-solid border-transparent;
	}

	.content {
		/* Row format for tags */
		@apply flex flex-wrap items-start;
	}

	.subcontent {
		@apply flex-grow w-0 p-1 m-1 rounded-sm;
		@apply text-center overflow-hidden text-xs text-ellipsis whitespace-nowrap select-none font-mono;
		@apply text-slate-200 bg-slate-800;
	}

	.subcontent:hover {
		@apply bg-slate-600;
	}

	.pivot:hover {
		@apply border-2 border-solid border-violet-600;
	}

	/* CSS for the tooltips */
	:global(.tippy-box[data-theme='pivot-tooltip']) {
		@apply py-1 px-1 text-xs font-mono rounded-lg shadow-sm;
		@apply text-white bg-violet-500;
	}

	:global(.tippy-box[data-theme='content-tooltip']) {
		@apply py-4 px-4 text-base font-mono rounded-lg shadow-sm;
		@apply text-white bg-violet-900;
	}
</style>

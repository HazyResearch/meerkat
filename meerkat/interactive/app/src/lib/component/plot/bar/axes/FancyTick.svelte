<script lang="ts">
	import Close from 'carbon-icons-svelte/lib/Close.svelte';
	import { createEventDispatcher, getContext } from 'svelte';
	import { createTippy } from 'svelte-tippy';
	import { followCursor } from 'tippy.js';

	const dispatch = createEventDispatcher();

	export let width: string;
	export let name: string;
	export let id: any;
	export let size: number;

	$: can_remove = getContext('can_remove');

	let tippy = (node: HTMLElement, parameters: any = null) => {};

	tippy = createTippy({
		placement: 'auto',
		allowHTML: true,
		theme: 'fancy-tick-tooltip',
		followCursor: 'vertical',
		plugins: [followCursor],
		duration: [0, 0],
		maxWidth: '95vw',
		interactive: true
	});
</script>

<div class="relative">
	<div
		class="bg-slate-100 z-10 rounded px-2 py-1 text-center"
		use:tippy={{ content: document.getElementById(`fancy-tick-${id}`)?.innerHTML }}
	>
		{name}
	</div>

	<div id="fancy-tick-{id}" class="hidden">
		<div class="grid grid-cols-[1fr_auto] items-center">
			{name}
			{#if can_remove}
				<div
					class="font-bold text-red-600 hover:text-red-800"
					on:click={() => dispatch('remove', id)}
				>
					<Close size=32/>
				</div>
			{/if}
		</div>
		<div class="mt-1 grid grid-flow-row grid-cols-2 space-x-1">
			<div class="font-bold">Count</div>
			<div>{size}</div>
			<div class="font-bold">ID</div>
			<div>{id}</div>
		</div>
	</div>
</div>

<style>
	/* CSS for the tooltip */
	:global(.tippy-box[data-theme='fancy-tick-tooltip']) {
		@apply py-2 px-4 text-base font-mono rounded shadow-md h-fit;
		@apply text-white bg-violet-500;
	}
</style>

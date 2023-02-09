<script lang="ts">
	import { Trash } from 'svelte-bootstrap-icons';
	import { getContext } from 'svelte';
	import { createTippy } from 'svelte-tippy';
	import { followCursor } from 'tippy.js';

	export let width: string;
	export let name: string;
	export let id: any;
	export let size: number;

	$: removeRow = getContext('removeRow');
	$: can_remove = getContext('can_remove');

	let tippy = (node: HTMLElement, parameters: any = null) => {};

	tippy = createTippy({
		placement: 'auto',
		allowHTML: true,
		theme: 'fancy-tick-tooltip',
		duration: [0, 0],
		interactive: true
	});
</script>

<div class="relative">
	<div
		class="bg-slate-100 z-10 rounded px-2 py-1 grid grid-cols-[1fr_auto]"
		use:tippy={{ content: document.getElementById(`fancy-tick-${id}`)?.innerHTML }}
	>
		<div class="whitespace-nowrap overflow-hidden text-ellipsis">
		{name}
		</div>

		{#if can_remove}
			<button
				class="font-bold text-slate-600 hover:bg-slate-300 rounded-sm p-1"
				on:click={() => removeRow(id)}
			>
				<Trash/>
		</button>
		{/if}
	</div>

	<div id="fancy-tick-{id}" class="hidden  overflow-hidden">
		<div class="grid grid-cols-[1fr_auto] items-center">
			{name}
			
		</div>
		<div class="mt-1 grid grid-flow-row grid-cols-2 space-x-1">
			<div class="font-bold">Count</div>
			<div>{size}</div>
			<div class="font-bold">ID</div>
			<div class="whitespace-nowrap overflow-hidden text-ellipsis">{id}</div>
		</div>
	</div>
</div>

<style>
	/* CSS for the tooltip */
	:global(.tippy-box[data-theme='fancy-tick-tooltip']) {
		@apply py-2 px-4 text-base font-mono rounded shadow-md h-fit;
		@apply bg-slate-100
	}
</style>

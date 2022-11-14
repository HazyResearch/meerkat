<script lang="ts">
	import Close from 'carbon-icons-svelte/lib/Close.svelte';
	import { createEventDispatcher } from 'svelte';
	import { getContext } from 'svelte';

	const dispatch = createEventDispatcher();

	export let width: string;
	export let name: string;
	export let id: any;
	export let size: number;

	$: can_remove = getContext('can_remove');

	let hover = false;
</script>

<div class="relative" on:mouseenter={() => (hover = true)} on:mouseleave={() => (hover = false)}>
	{#if hover}
		<div class="absolute top-0 left-0 bg-slate-100 z-40 rounded px-2 py-1 shadow-md h-fit">
			<div class="grid grid-cols-[1fr_auto]">
				<div>
					{name}
				</div>
				{#if can_remove}
					<div
						class="font-bold text-red-400 hover:text-red-600"
						on:click={() => dispatch('remove', id)}
					>
						<Close />
					</div>
				{/if}
			</div>
			<div class="grid grid-flow-row grid-cols-2 space-x-1">
				<div class="font-bold">Name</div>
				<div>{name}</div>
				<div class="font-bold">Count</div>
				<div>{size}</div>
				<div>ID</div>
				<div>{id}</div>
			</div>
		</div>
	{/if}

	<div class="bg-slate-100 z-10 rounded px-2 py-1 w-fit">
		{name}
	</div>
</div>

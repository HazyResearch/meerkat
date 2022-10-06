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

<div
	class="flex items-center h-full text-xs whitespace-nowrap overflow-hidden hover:overflow-visible space-x-1"
	style="
        width: {hover ? '100%' : width};
        "
	on:mouseenter={() => (hover = true)}
	on:mouseleave={() => (hover = false)}
>
	{#if can_remove}
		<div
			class="text-violet-400 border rounded mr-1 bg-red-100 hover:bg-red-200"
			on:click={() => dispatch('remove', id)}
		>
			<Close />
		</div>
	{/if}
 	<div class="font-mono rounded-sm bg-slate-200 px-0.5">
		{size}
	</div>
	<div class={hover ? 'bg-slate-200' : 'bg-inherit'}>
		{name} 
	</div>
</div>

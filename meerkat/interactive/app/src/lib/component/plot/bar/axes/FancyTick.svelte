<script lang="ts">
	import Close from 'carbon-icons-svelte/lib/Close.svelte';
	import { createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher();

	export let width: string;
	export let name: string;
	export let size: number;

	let hover = false;
</script>

<div
	class="flex items-center h-full text-xs whitespace-nowrap font-mono text-violet-400 overflow-hidden hover:overflow-visible"
	style="
        width: {hover ? '100%' : width};
        "
	on:mouseenter={() => (hover = true)}
	on:mouseleave={() => (hover = false)}
>
	<div class="text-violet-400 border rounded mr-1 bg-red-100 hover:bg-red-200" on:click={() => dispatch('remove')}>
		<Close />
	</div>
	<div class={hover ? 'bg-slate-200' : 'bg-inherit'}>
		{name} ({size})
	</div>
</div>

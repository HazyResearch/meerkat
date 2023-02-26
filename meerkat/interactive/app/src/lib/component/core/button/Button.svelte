<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { Nullable } from 'vitest';
	import Icon from '$lib/component/core/icon/Icon.svelte';

	export let title: string = '';
	export let icon: string | Nullable = null;
	export let classes: string = "";

	let dispatch = createEventDispatcher();

	function dispatchClick() {
		dispatch('click', {});
	}
</script>

<div class={'w-fit bg-slate-100 py-1 rounded-md flex items-center text-slate-600 hover:bg-slate-200' + classes}>
	<button
		on:click={(e) => {
			dispatchClick();
		}}
	>
		<div class="flex flex-row gap-1 items-center">
			{#if icon}
				<Icon name={icon} />
			{/if}
			{title}
		</div>
	</button>
</div>

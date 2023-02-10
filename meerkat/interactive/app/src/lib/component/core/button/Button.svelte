<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let title: string = '';
	export let classes: string = 'bg-slate-100 py-3 rounded-lg drop-shadow-md flex flex-col hover:bg-slate-200';

	let dispatch = createEventDispatcher();

	function dispatchClick() {
		dispatch('click', {});
	};
</script>

<div class={classes}>
	<button
		on:click={(e) => {
			dispatchClick();
		}}>{title}</button
	>
</div>

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	let dispatch = createEventDispatcher();

	export let value: boolean = false;
</script>

<label class="relative inline-flex items-center cursor-pointer w-fit">
	<input
		type="checkbox"
		value=""
		class="sr-only peer"
		on:change={(e) => {
			dispatch('change', { value: e.target.checked });
		}}
		bind:checked={value}
	/>
	<div
		class="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-purple-300 dark:peer-focus:ring-purple-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-purple-600"
	/>
</label>

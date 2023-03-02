<script lang="ts">
    export let label_left: string;
    export let label_right: string;
	export let checked: boolean;
</script>

<label class="inline-flex relative items-center mr-5 cursor-pointer">
	<span class="label mr-3">
		{#if label_left}{label_left}{/if}
	</span>
	<input type="checkbox" class="sr-only peer" bind:checked />
	<div class="toggle peer" />
	<span class="label ml-3">
		{#if label_right}{label_right}{/if}
	</span>
</label>

<style>
	.label {
		@apply text-sm font-medium text-gray-300 dark:text-gray-300;
	}

	.toggle {
		@apply w-11 h-6 rounded-full;
		@apply peer-checked:after:translate-x-full peer-checked:after:border-white peer-checked:bg-purple-600;
		@apply bg-gray-200  dark:bg-purple-600  dark:border-gray-600;
		@apply after:content-[''] after:absolute after:top-0.5 after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all;
	}
</style>

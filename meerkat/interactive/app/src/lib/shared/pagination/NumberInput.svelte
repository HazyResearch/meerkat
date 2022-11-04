<script lang="ts">
	export let value: number;
	export let min: number = 1;
	export let max: number = Infinity;
	export let placeholder_text: string = 'Enter a number';
	export let hint_text: string = 'Text';
</script>

<div class="flex w-40">
	<div
		class="inline-flex justify-center w-3/5 p-2.5 border border-solid text-sm rounded-l-lg dark:bg-gray-800 dark:text-gray-100"
	>
		{hint_text}
	</div>
	<input
		class="inline-flex w-2/5 p-2.5 border border-solid text-sm dark:hover:border-violet-600 dark:focus:border-violet-600 dark:bg-gray-600 dark:text-gray-100"
		bind:value
		type="number"
		placeholder={placeholder_text}
		{min}
		{max}
	/>
</div>

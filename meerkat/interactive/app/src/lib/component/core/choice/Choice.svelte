<script lang="ts">
	import Select from 'svelte-select';
	import { createEventDispatcher } from 'svelte';

	let dispatch = createEventDispatcher();

	/** Array of choices to display in the dropdown. */
	export let choices: Array<any>;
	/** The current value of the dropdown. */
	export let value: any;
	/** The type of GUI to use. Either `dropdown` or `radio`. */
	export let gui_type: 'dropdown' | 'radio' = 'dropdown';
	/** The title to display above the dropdown. */
	export let title: string = '';

	let dispatchSelect = (index: number) => {
		dispatch('select', { index: index });
	};
</script>

<div
	class="w-full flex space-x-2 items-center bg-slate-100 py-1 rounded-lg px-2 drop-shadow-md z-20"
>
	{#if title != ''}
		<div class="text-center text-sm font-bold text-slate-600">
			{title}
		</div>
	{/if}

	{#if gui_type == 'dropdown'}
		<div class="themed flex-grow">
			<Select
				id="column"
				{value}
				items={choices}
				on:change={(e) => {
					console.log(e);
					dispatchSelect(e.detail.index);
				}}
			/>
		</div>
	{:else if gui_type == 'radio'}
		{#each choices as choice, i}
			<div class="themed flex items-center justify-center">
				<input
					id="default-radio-{i}"
					type="radio"
					value=""
					name="default-radio"
					class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
					on:click={(e) => {
						dispatchSelect(i);
					}}
				/>
				<label
					for="default-radio-{i}"
					class="ml-2 text-sm font-medium text-gray-900 dark:text-gray-300">{choice}</label
				>
			</div>
		{/each}
	{/if}
</div>

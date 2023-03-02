<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let values: Array<string | number>;
	export let labels: Array<string | number>;
	export let value: string | number;
	export let disabled: boolean = false;
	export let classes: string = '';

	// Make an `items` array from the `values` and `labels` arrays.
	// It should contain a list of objects with `value` and `name` properties.
	$: items = values.map((value, index) => {
		return {
			value: value,
			name: labels[index]
		};
	});

	const dispatch = createEventDispatcher();
</script>

<select
	on:change={(e) => dispatch('change', { value })}
	bind:value
	class="rounded-md bg-slate-100 border-0 px-3 text-slate-600 w-full"
>
	{#each items as item}
		<option value={item.value}>{item.name}</option>
	{/each}
</select>

<!-- <div class={classes}>
	<Select
		on:change={(e) => dispatch('change', { value })}
		bind:items
		bind:value
		bind:disabled
		defaultClass="bg-purple-200 border-purple-500 ring-purple-200 ring-offset-0 ring-0 focus:border-purple-500 focus:ring-purple-200 focus:ring-offset-0 focus:ring-0 p-2 rounded-lg w-fit"
	/>
</div> -->

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
	class="rounded-md bg-slate-100 border-0 px-3 text-slate-600"
>
	{#each items as item}
		<option value={item.value}>{item.name}</option>
	{/each}
</select>



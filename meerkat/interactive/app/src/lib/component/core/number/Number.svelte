<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let dtype: string = 'auto';
	export let precision: number = 3;
	export let percentage: boolean = false;
	export let editable: boolean = false;
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');

	let valid = true;

	if (dtype === 'auto') {
		// The + operator converts a string or number to a number if possible
		if (typeof +data === 'number') {
			data = +data;
			if (Number.isInteger(data)) {
				dtype = 'int';
			} else {
				dtype = 'float';
			}
		} else {
			dtype = 'string';
		}
	}

	if (dtype === 'float') {
		if (percentage) {
			data = (data * 100).toPrecision(precision) + '%';
		} else if (!isNaN(data)) {
			data = data.toPrecision(precision).toString();
		} else {
			data = 'NaN';
		}
	}
</script>

{#if editable}
	<input
		class={'input w-full' + (valid ? '' : ' outline-red-500')}
		bind:value={data}
		on:input={() => (valid = data === '' || data === 'NaN' || !isNaN(+data))}
		on:change={() => cellEdit(dtype === 'string' ? data : +data)}
	/>
{:else}
	<div class={classes}>
		{data}
	</div>
{/if}

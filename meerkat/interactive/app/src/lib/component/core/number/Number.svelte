<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let dtype: string = 'auto';
	export let precision: number = 3;
	export let percentage: boolean = false;
	export let editable: boolean = false;
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');

	let prevData: any = data;
	let dtypeBefore: string = dtype;

	if (dtype === 'auto') {
		// The + operator converts a string or number to a number if possible
		// if (typeof +data === 'number') {
		if (typeof data === 'number') {
			// data = +data;
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
		if (typeof data === 'string') {
			data = +data;
		}
		if (percentage) {
			data = (data * 100).toPrecision(precision) + '%';
		} else if (!isNaN(data)) {
			console.log(data, typeof data, dtypeBefore);
			data = data.toPrecision(precision).toString();
		} else {
			data = 'NaN';
		}
	}

	function validator(node, value: any) {
		return {
			update(value: any) {
				data = value === null || typeof +value !== 'number' ? prevData : +value;
				prevData = data;
			}
		};
	}
</script>

{#if editable}
	<!-- type="number" -->
	<input
		class="input w-full"
		on:change={() => {
			cellEdit(dtype === 'string' ? data : +data);
		}}
		use:validator={data}
		bind:value={data}
	/>
{:else}
	<div class={classes}>
		{data}
	</div>
{/if}

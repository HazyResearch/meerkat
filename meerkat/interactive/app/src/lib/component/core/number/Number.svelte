<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let dtype: string = 'auto';
	export let precision: number = 3;
	export let percentage: boolean = false;
	export let editable: boolean = false;
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');

	let invalid = false;
	let showInvalid = false;

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
		class={'input w-full' + (invalid && ' outline-red-500')}
		class:invalid={showInvalid}
		bind:value={data}
		on:input={() => (invalid = data !== '' && data !== 'NaN' && isNaN(+data))}
		on:change={() => {
			if (invalid) {
				showInvalid = true;
				setTimeout(() => (showInvalid = false), 1000);
			} else {
				cellEdit(dtype === 'string' ? data : +data);
			}
		}}
	/>
{:else}
	<div class={classes}>
		{data}
	</div>
{/if}

<style>
	@keyframes shake {
		0% {
			margin-left: 0rem;
		}
		25% {
			margin-left: 0.5rem;
		}
		75% {
			margin-left: -0.5rem;
		}
		100% {
			margin-left: 0rem;
		}
	}

	.invalid {
		animation: shake 0.2s ease-in-out 0s 2;
	}
</style>

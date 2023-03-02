<script lang="ts">
	import { Checkbox } from 'flowbite-svelte';
	import type { FormColorType } from 'flowbite-svelte/types';
	import { createEventDispatcher } from 'svelte';
	import { getContext } from 'svelte/internal';

	/** The checked state of the checkbox. */
	export let checked: boolean = false;

	/** Whether the checkbox is disabled. */
	export let disabled: boolean = false;

	/** Color of the checkbox. */
	export let color: FormColorType = 'purple';

	/** The classes to apply to the container element. */
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');
	const dispatch = createEventDispatcher();
</script>

<div class={classes}>
	<Checkbox
		on:change={(e) => {
			cellEdit(checked);
			dispatch('change');
		}}
		bind:checked
		bind:disabled
		bind:color
	>
		<slot />
	</Checkbox>
</div>

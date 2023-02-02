<script lang="ts">
	import { Checkbox } from 'flowbite-svelte';
	import type { FormColorType } from 'flowbite-svelte/types';
	import { createEventDispatcher } from 'svelte';

	/** The checked state of the checkbox. */
	export let checked: boolean = false;

	/** Whether the checkbox is disabled. */
	export let disabled: boolean = false;

    /** Color of the checkbox. */
    export let color: FormColorType = 'purple';

	/** The classes to apply to the container element. */
	export let classes: string = 'bg-violet-50 p-2 rounded-lg';

	const dispatch = createEventDispatcher();
</script>

<div class={classes}>
	<Checkbox
		on:change={(e) => dispatch('change', { detail: e.detail })}
		bind:checked
		bind:disabled
		bind:color
	>
		<slot />
	</Checkbox>
</div>

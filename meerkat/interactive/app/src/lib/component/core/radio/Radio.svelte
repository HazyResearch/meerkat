<script lang="ts">
	import { Radio } from 'flowbite-svelte';
	import type { FormColorType } from 'flowbite-svelte/types';
	import { createEventDispatcher } from 'svelte';

    /** The name of the radio group. */
    export let name: string = '';

    /** The value of the radio button. */
    export let value: string = '';

	/** Whether the checkbox is disabled. */
	export let disabled: boolean = false;

    /** Color of the checkbox. */
    export let color: FormColorType = 'purple';

	/** The classes to apply to the container element. */
	export let classes: string = 'bg-violet-50 p-2 rounded-lg w-fit';

	const dispatch = createEventDispatcher();
</script>

<div class={classes}>
	<Radio
		on:change={(e) => dispatch('change', { detail: e.detail })}
        bind:name
        bind:value
		bind:disabled
		bind:color
	>
		<slot />
    </Radio>
</div>

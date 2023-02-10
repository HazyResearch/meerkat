<script lang="ts">
	import { Radio } from 'flowbite-svelte';
	import type { FormColorType } from 'flowbite-svelte/types';
	import { createEventDispatcher } from 'svelte';

	/** The values of the radio buttons. */
	export let values: Array<string>;

    /** Selected index of the radio group. */
    export let selected: number | undefined = undefined;
    let group = selected !== undefined ? values[selected] : undefined;

	/** Whether the radio is disabled. */
	export let disabled: boolean = false;

	/** Color of the radio. */
	export let color: FormColorType = 'purple';

    /** Orientation of the radio group. */
    export let horizontal: boolean = true;

	/** The classes to apply to the container element. */
	export let classes: string = 'bg-violet-50 p-2 rounded-lg w-fit';
    classes += horizontal ? ' flex flex-row space-x-4' : ' flex flex-col space-y-2';

	console.log("selected,", selected)
	const dispatch = createEventDispatcher();
</script>

<div class={classes}>
	{#each values as value, i}
		<Radio
			on:change={e => {selected = i; dispatch('change', { index: i })}}
			name="radio-group"
			bind:value
			bind:disabled
			bind:color
            bind:group
            class="{disabled ? 'text-gray-400' : 'text-purple-500'}"
			checked={selected === i}
		>   
			{value}
		</Radio>
	{/each}
</div>

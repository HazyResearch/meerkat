<script lang="ts">
	// import { Range } from 'flowbite-svelte';
	import Range from './Range.svelte';
	import { createEventDispatcher } from 'svelte';

	/** The value of the slider. */
	export let value: number = 0;

	/** The minimum value of the slider. */
	export let min: number = 0;

	/** The maximum value of the slider. */
	export let max: number = 100;

	/** The step of the slider. */
	export let step: number = 1;

	/** Whether the slider is disabled. */
	export let disabled: boolean = false;

	/** The classes to apply to the container element. */
	export let classes: string = 'bg-violet-50 p-2 rounded-lg w-fit';

	const dispatch = createEventDispatcher();
</script>

<div class={classes}>
	<div class="flex flex-col">
        <!-- Make sure the right end of the div doesn't fall off -->
        <div class="-ml-2 px-2 w-full">
			<p
				class="text-sm text-purple-500"
				style={`margin-left: ${((value - min) / (max - min)) * 100}%`}
			>
				{value}
			</p>
		</div>
		<Range
			on:change={(e) => dispatch('change', { value })}
			bind:value
			bind:min
			bind:max
			bind:step
			bind:disabled
			classes="accent-purple-500"
		/>
		<div class="flex">
			<div class="flex-1 text-sm text-purple-400">{min}</div>
			<div class="flex-1 text-sm text-purple-400 text-right">{max}</div>
		</div>
	</div>
</div>

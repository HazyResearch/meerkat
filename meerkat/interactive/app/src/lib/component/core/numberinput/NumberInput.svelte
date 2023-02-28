<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	/** Value in the input field. */
	export let value: number;

	/** Placeholder text. */
	export let placeholder: string = 'Enter a number...';

	/** Debounce timer (in ms). */
	export let debounceTimer: number = 150;
	export let classes: string = 'grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400';

	let timer: any;
	const debounce = (e: KeyboardEvent) => {
		clearTimeout(timer);
		timer = setTimeout(() => {
			value = e.target.value;
			// let the value update before dispatching the event.
			// we pick 3ms as a magic number, but it should be enough
			// to let the value update.
			setTimeout(() => {
				if (e.code == 'Enter') {
					dispatch('keyenter', { value: value });
				}
				dispatch('keyup', e);
			}, 3);
		}, debounceTimer);
	};

	let initialValue: number;
	$: initialValue = value;

	const dispatch = createEventDispatcher();
</script>

<input
	type="number"
	{placeholder}
	value={initialValue}
	class={classes}
	on:keyup={(e) => debounce(e)}
	on:keypress
	on:blur={(e) => {
		dispatch('blur', { value: value });
	}}
/>

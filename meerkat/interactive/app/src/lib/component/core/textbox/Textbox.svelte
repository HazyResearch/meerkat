<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	/** Text in the textbox. */
	export let text: string;

	/** Placeholder text. */
	export let placeholder: string = 'Write some text...';

	/** Debounce timer (in ms). */
	export let debounceTimer: number = 150;
	export let classes: string = 'grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400';

	let timer: any;
	const debounce = (e: KeyboardEvent) => {
		clearTimeout(timer);
		timer = setTimeout(() => {
			console.log('debounding');
			text = e.target.value;
			// let the value update before dispatching the event.
			// we pick 3ms as a magic number, but it should be enough
			// to let the value update.
			setTimeout(() => {
				if (e.code == 'Enter') {
					dispatch('keyenter', { text: text });
				}
				dispatch('keyup', e);
			}, 3);
		}, debounceTimer);
	};

	let initialText: string;
	$: initialText = text;

	const dispatch = createEventDispatcher();
</script>

<input
	type="text"
	{placeholder}
	value={initialText}
	class={classes}
	on:keyup={(e) => debounce(e)}
	on:keypress
	on:blur={(e) => {
		dispatch('blur', { text: text });
	}}
/>

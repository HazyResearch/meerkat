<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	/** Text in the textbox. */
	export let text: string;

	/** Placeholder text. */
	export let placeholder: string = 'Write some text...';

	/** Debounce timer (in ms). */
	export let debounceTimer: number = 150;

	let timer: any;
	const debounce = (e: KeyboardEvent) => {
		clearTimeout(timer);
		timer = setTimeout(() => {
			console.log('debounding');
			text = e.target.value;
			dispatch('keyup', e);
		}, debounceTimer);
	};

	const initialText = text;

	const dispatch = createEventDispatcher();
</script>

<input
	type="text"
	{placeholder}
	value={initialText}
	class="grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400"
	on:keyup={(e) => debounce(e)}
	on:keypress
	on:blur={(e) => {
		dispatch('blur', { text: text });
	}}
/>

<script lang="ts">
	/** Text in the textbox. */
	export let text: string;

	/** Placeholder text. */
	export let placeholder: string = "Write some text...";

	/** Debounce timer (in ms). */
	export let debounceTimer: number = 150;

	let timer: any;
	const debounce = (v: string) => {
		clearTimeout(timer);
		timer = setTimeout(() => {
			text = v;
		}, debounceTimer);
	};

	const initialText = text;
</script>

<input
	type="text"
	{placeholder}
	value={initialText}
	class="grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400"
	on:keyup={({ target: { value } }) => debounce(value)}
	on:keypress
/>

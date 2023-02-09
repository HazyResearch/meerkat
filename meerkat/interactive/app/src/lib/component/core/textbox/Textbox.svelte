<script lang="ts">
	/** Text in the textbox. */
	export let text: string;

	let timer: any;
	const debounce = (v: string) => {
		clearTimeout(timer);
		timer = setTimeout(() => {
			text = v;
		}, 150);
	};
</script>

<input
	type="text"
	placeholder="Write some text..."
	class="grow h-10 px-3 rounded-md shadow-md"
	on:keyup={({ target: { value } }) => debounce(value)}
	on:keypress
/>

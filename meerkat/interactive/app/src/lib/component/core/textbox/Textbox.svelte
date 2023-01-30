<script lang="ts">
	/** Text in the textbox. */
	export let text: string;

	/** Title the textbox to help the user. */
	export let title: string = '';

	let timer: any;
	const debounce = (v: string) => {
		clearTimeout(timer);
		timer = setTimeout(() => {
			text = v;
		}, 300);
	};
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md flex flex-col">
	{#if title != ''}
		<div class="font-bold text-xl text-slate-600 self-start pl-2">
			{title}
		</div>
	{/if}
	<div class="px-2 w-full flex items-center">
		<input
			type="text"
			placeholder="Write some text..."
			class="grow h-10 px-3 rounded-md shadow-md"
			on:keyup={({ target: { value } }) => debounce(value)}
		/>
	</div>
</div>

<script lang="ts">
	import type { Writable } from 'svelte/store';
	import Select from 'svelte-select';

	export let choices: Writable;
	export let value: Writable<string>;
	export let gui_type: string;
	export let title: string = '';

	function handleSelect(event) {
		console.log("Selected", event.detail.value);
		$value = event.detail.value;
	}
</script>

<div class="w-full flex space-x-2 items-center bg-slate-100 py-1 rounded-lg px-2 drop-shadow-md z-20">
	{#if title != ''}
		<div class="text-center text-sm font-bold text-slate-600">
			{title}
		</div>
	{/if}
	<div class="themed flex-grow">
		<Select
			id="column"
			value={$value}
			items={$choices}
			showIndicator={true}
			listPlacement="auto"
			on:select={handleSelect}
		/>
	</div>
</div>

<script lang="ts">
	import type { Writable } from 'svelte/store';
	import { getContext } from 'svelte';
	import Select from 'svelte-select';

	export let choices: Writable;
	export let value: Writable<string>;
	export let gui_type: string;

	function handleSelect(event) {
		$value = event.detail.value;
	}

</script>

<div class="w-full flex items-center bg-slate-100 py-1 rounded-lg drop-shadow-md z-20">
	<div class="themed px-2 flex-grow">
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

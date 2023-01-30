<script lang="ts">
	import Select from 'svelte-select';
	import type { Writable } from 'svelte/store';

	export let choices: Writable<Array<any>>;
	export let selected: Writable<Array<any>>;
	export let title: string = '';

    // Make a selected_detail variable that is bound to the Select component
    let selected_detail = $selected;
    // Make a reactive statement that updates selected, when selected_detail changes
    $: $selected = selected_detail.map((item: any) => item.value);
</script>

<div
	class="w-full flex space-x-2 items-center bg-slate-100 py-1 rounded-lg px-2 drop-shadow-md z-20"
>
	{#if title != ''}
		<div class="text-center text-sm font-bold text-slate-600">
			{title}
		</div>
	{/if}
	<div class="themed flex-grow">
        <!-- bind:value to selected_detail, because value contains a list of objects,
        each object has the value key, along with other keys -->
		<Select
			id="column"
			bind:value={selected_detail}
			items={$choices}
			multiple={true}
		/>
	</div>
</div>

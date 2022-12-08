<script lang="ts">
	import type { Writable } from 'svelte/store';
    // import MultiSelect from './_MultiSelect.svelte';

	export let choices: Writable;
	export let selected: Writable<Array<string>>;
	export let title: string = '';
</script>

<div class="w-full flex space-x-2 items-center bg-slate-100 py-1 rounded-lg px-2 drop-shadow-md z-20">
	{#if title != ''}
		<div class="text-center text-sm font-bold text-slate-600">
			{title}
		</div>
	{/if}
    <select name="cars" id="cars" multiple>
        {#each $choices as choice}
            <option value={choice} on:click={() => {
                if ($selected.includes(choice)) {
                    $selected = $selected.filter((item) => item !== choice);
                } else {
                    $selected.push(choice);
                }
                console.log($selected);
            }}>{choice}</option>
        {/each}
    </select>
</div>

<!-- <MultiSelect value={[]} bind:selected>
    {#each choices as choice}
        <option value={choice}>{choice}</option>
    {/each}
</MultiSelect> -->

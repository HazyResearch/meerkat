<script lang="ts">
	import { getContext } from 'svelte/internal';
	import Markdown from '../markdown/Markdown.svelte';

	export let data: any;
	export let view: string = 'line';
	export let editable: boolean = false;

	const cellEdit: CallableFunction = getContext('cellEdit');
</script>

{#if view === 'line'}
	{#if editable}
		<input
			class="input input-bordered grow h-7 px-3 rounded-md shadow-md"
			on:change={() => {
				cellEdit(data);
			}}
			bind:value={data}
		/>
	{:else}
		<div class="text-ellipsis whitespace-nowrap overflow-hidden">
			{data}
		</div>
	{/if}
{:else if view === 'wrapped'}
	<div class="whitespace-pre-line">
		<Markdown body={data} />
	</div>
{/if}

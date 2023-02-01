<script lang="ts">
	import {createEventDispatcher} from 'svelte';

	export let data: any;
	export let view: string = 'line';
	export let editable: boolean = true;

	const dispatch = createEventDispatcher();
	const edit = () => {dispatch('edit', {value: data})}

</script>

{#if view === 'line'}
	{#if editable}
		<input 				
			class="input input-bordered grow h-7 px-3 rounded-md shadow-md"
			on:change={edit} 
			bind:value={data} 
		/>
	{:else}
		{data}
	{/if}
{:else}
else
{/if}
<!-- 
{#if view === 'logo'}
	<span><Globe2 /></span>
{:else if view === 'thumbnail'}
	<div
		class="bg-white flex h-full w-full aspect-video content-center items-center rounded-md shadow-md border-black text-center"
	>
		{data}
	</div>
{:else}
	<div class="h-full w-full rounded-md shadow-md border-black">
		<iframe
			srcdoc={data}
			title={'title'}
			class="rounded-md"
			frameborder="0"
			style="height: 100%; width: 100%;"
		/>
	</div>
{/if} -->

<!-- {#if false}
	<div
		class="bg-white flex h-full w-full aspect-video content-center items-center rounded-md shadow-md border-black text-center"
	>
		{data}
	</div>
{:else}
	{data}
{/if} -->

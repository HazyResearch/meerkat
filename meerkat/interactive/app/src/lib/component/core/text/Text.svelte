<script lang="ts">
	import {getContext} from 'svelte/internal';
	import Markdown from '@magidoc/plugin-svelte-marked';


	export let data: any;
	export let view: string = 'line';
	export let editable: boolean = false;
	export let markdown: boolean = true;

	const cellEdit: CallableFunction = getContext('cellEdit');
</script>


{#if view === 'line'}
	{#if editable}
		<input 				
			class="input input-bordered grow h-7 px-3 rounded-md shadow-md"
			on:change={() => {cellEdit(data)}} 
			bind:value={data} 
		/>
	{:else}
	<div class="text-ellipsis whitespace-nowrap overflow-hidden">
		{data}
	</div>
	{/if}
{:else if view === 'wrapped'}
	<div class="whitespace-pre-line">
		<Markdown source={data} />
	</div>
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

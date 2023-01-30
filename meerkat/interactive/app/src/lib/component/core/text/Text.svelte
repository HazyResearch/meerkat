<script lang="ts">
	export let data: any;
	export let view: string = 'full';
</script>

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
{/if}

<!-- {#if false}
	<div
		class="bg-white flex h-full w-full aspect-video content-center items-center rounded-md shadow-md border-black text-center"
	>
		{data}
	</div>
{:else}
	{data}
{/if} -->

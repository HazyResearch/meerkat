<script lang="ts">
	import Cell from '$lib/components/cell/Cell.svelte';

	export let header: string | undefined = undefined;
	export let content: any = undefined;
	export let layout: string;
</script>

<div class="pill">
    {#if header}
        <div 
            class="header header-text" 
            class:w-full={layout === 'wide-header' || layout === 'equal'}
            class:rounded-md={!content}
        >
            <Cell data={header} />
        </div>
    {/if}
    {#if content}
        <div 
            class="content content-text" 
            class:w-full={layout === 'wide-content' || layout === 'equal'}
            class:rounded-md={!header}
        >
            <Cell data={content} />
        </div>
    {/if}
</div>

<style>
	.pill {
		@apply flex;
	}

	.header-text,
	.content-text {
		@apply text-center text-xs text-ellipsis whitespace-nowrap select-none font-mono;
		@apply text-slate-200;
	}
	.content-text {
		@apply overflow-hidden text-right;
	}

	.header {
		@apply p-1 mr-0 ml-0 m-1 rounded-l-md;
		@apply bg-red-800;
	}

	.content {
		@apply p-1 pl-2 ml-0 m-1 rounded-r-md;
		@apply bg-slate-500;
	}
</style>

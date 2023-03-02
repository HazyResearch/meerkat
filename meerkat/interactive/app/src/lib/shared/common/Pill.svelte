<script lang="ts">
	import Cell from '$lib/shared/cell/Cell.svelte';

	export let header: string | undefined = undefined;
	export let content: any = undefined;
	export let layout: "wide-header" | "equal" | "wide-content";
</script>

<div class="flex h-fit">
    {#if header}
        <div 
            class="p-1 mr-0 ml-0 m-1 rounded-l-md header-text text-violet-800 bg-slate-200 text-center text-xs text-ellipsis font-bold whitespace-nowrap select-none"
            class:w-full={layout === 'wide-header' || layout === 'equal'}
            class:rounded-md={!content}
        >
            {header}
        </div>
    {/if}
    {#if typeof(content) !== "undefined"}
        <div 
            class="p-1 pl-2 ml-0 m-1 rounded-r-md bg-slate-200 overflow-hidden text-right text-center text-xs text-ellipsis whitespace-nowrap select-none font-mono" 
            class:w-full={layout === 'wide-content' || layout === 'equal'}
            class:rounded-md={!header}
        >
            <Cell data={content} cellComponent="basic"/>
        </div>
    {/if}
</div>



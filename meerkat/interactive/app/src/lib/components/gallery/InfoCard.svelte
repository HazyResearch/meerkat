<script lang="ts">    
    import Item from '$lib/components/item/Item.svelte';

    export let pivot: any;
    export let content: any;

    // Give the card the `flex-grow` Tailwind class to horizontally 
    // fill out space in the (containing) flex container.
    export let card_flex_grow: boolean = false;
    
</script>

<div class="card" class:flex-grow={card_flex_grow}>
    <div class="pivot">
        <Item data={pivot} />
    </div>
    <div class="content">
        {#each content as subcontent, j}
            <div class="subcontent">
                <Item data={subcontent}/>
            </div>
        {/each}
    </div>
</div>


<style>
    .card {
        min-width: var(--card-width, "");
        @apply bg-slate-200;
        @apply p-4 rounded-lg;
	}

    .pivot {
        @apply self-center;
    }
    
    .content {
		@apply flex flex-wrap items-start;
	}

	.subcontent {
        @apply flex-grow w-0 p-1 m-1 rounded-sm;
		@apply text-center overflow-hidden text-xs text-ellipsis whitespace-nowrap select-none font-mono;
        @apply text-slate-200 bg-slate-800;
	}
</style>
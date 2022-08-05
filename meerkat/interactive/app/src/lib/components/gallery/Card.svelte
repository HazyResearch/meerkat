<script lang="ts">
    import { tippy, createTippy } from 'svelte-tippy';
    import {followCursor} from 'tippy.js';

    import Item from '$lib/components/item/Item.svelte';

    export let id: string;
    export let pivot: any;
    export let content: any;
    export let content_type: string;
    export let layout: string;

    // Give the card the `flex-grow` Tailwind class to horizontally 
    // fill out space in the (containing) flex container.
    export let card_flex_grow: boolean = false;
    
    // Tooltip setup
    export let pivot_tooltip: boolean = true;
    export let content_tooltip: boolean = true;
    
    let pivot_tippy = (node: HTMLElement, parameters: any = null) => {};
    if (pivot_tooltip) {
        pivot_tippy = createTippy({
            allowHTML: true,
            theme: 'pivot-tooltip',
            trigger: 'click',
            duration: [0, 0],
            maxWidth: '95vw',
        });
    };

    let content_tippy = (node: HTMLElement, parameters: any = null) => {};
    if (content_tooltip) {
        content_tippy = createTippy({
            allowHTML: true,
            followCursor: true,
            plugins: [followCursor],
            theme: 'content-tooltip',
            duration: [0, 0],
            maxWidth: '95vw',
        });
    };

    
</script>

<div 
    class="card" 
    class:flex-grow={card_flex_grow} 
    class:card-masonry={layout === 'masonry'}
    class:card-gimages={layout === 'gimages'}
>
    <div 
        class="pivot" 
        class:pivot-masonry={layout === 'masonry'}
        class:pivot-gimages={layout === 'gimages'}
        use:pivot_tippy={{content: document.getElementById(`${id}-pivot-tooltip`)?.innerHTML}}
    >
        <!-- The pivot item, followed by the tooltip content. -->
        <Item data={pivot} />
        {#if pivot_tooltip}
            <div id="{id}-pivot-tooltip" class="hidden">
                <slot name="pivot-tooltip">
                    <Item data={pivot} />
                </slot>
            </div>
        {/if}
    </div>
    <div class="content">
        {#each content as subcontent, j}
            <div 
                class="subcontent" 
                use:content_tippy={{content: document.getElementById(`${id}-content-tooltip-${j}`)?.innerHTML}}
            >
                <Item data={subcontent}/>
                {#if content_tooltip}
                    <div id="{id}-content-tooltip-{j}" class="hidden">
                        <Item data={subcontent} />
                    </div>
                {/if}
            </div>
            
        {/each}
    </div>
</div>


<style>
    .card {
        min-width: var(--card-width, "");
	}

    .card-masonry {
        /* Solution 1: multiple columns in a masonry layout */
		@apply m-2 break-inside-avoid h-auto;
    }

    .card-gimages {
        /* Solution 2: flex containers in the Google Images style */
        @apply m-1;
        @apply rounded-lg border shadow-md;
        @apply dark:bg-gray-700 dark:border-gray-600;
        @apply flex flex-col;
    }
    
    .pivot {
        @apply self-center;
    }

    .pivot-masonry {
        
    }

    .pivot-gimages {
        height: var(--pivot-height, 20vh);
        @apply self-center;
    }
    
    .content {
        /* Row format for tags */
		@apply flex flex-wrap items-start;
	}

	.subcontent {
        @apply flex-grow w-0 p-1 m-1 rounded-sm;
		@apply text-center overflow-hidden text-xs text-ellipsis whitespace-nowrap select-none font-mono;
        @apply text-slate-200 bg-slate-800;
	}

	.subcontent:hover {
        @apply bg-slate-600;
	}

    .card:hover > .pivot {
       filter: blur(2px);
    }

    /* CSS for the tooltips */
    :global(.tippy-box[data-theme='pivot-tooltip']) {
    }

    :global(.tippy-box[data-theme='content-tooltip']) {
        @apply py-4 px-4 text-base font-mono rounded-lg shadow-sm;
        @apply text-white bg-gray-900;
    }
</style>
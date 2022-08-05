<script lang="ts">
	import Card from './Card.svelte';
    import { Slider } from "carbon-components-svelte";
    import InfoCard from './InfoCard.svelte';

	export let column_infos: Array<string> = [];
    export let rows: Array<any> = [];
        
    export let layout = 'gimages';
    
    // Main column to display.
    export let main_column: string = 'image';
    
    // Columns to use for tags in the GalleryView.
    export let tag_columns: Array<string> = [];
                
    let columns = column_infos.map((col: any) => col.name);
    let column_components = column_infos.map((col: any) => col.cell_component);

	let tag_indices: Array<number> = tag_columns.map((tag) => columns.indexOf(tag));
	let main_index: number = columns.indexOf(main_column);
    let pivot_cell_component = column_components[main_index];

    let pivot_height = 20;
</script>

<svelte:head>
	<link rel="stylesheet" href="https://unpkg.com/carbon-components-svelte/css/g90.css" />
</svelte:head>

<div class="h-full overflow-hidden">
    <div class="m-2">
        <Slider 
            fullWidth
            labelText="Image height"
            min={1}
            max={95}
            minLabel="1%"
            maxLabel="95%"
            bind:value={pivot_height}
        />
    </div>
    <div
        class="panel overflow-y-auto"
        class:panel-masonry={layout === 'masonry'}
        class:panel-gimages={layout === 'gimages'}
        style:height={layout === 'gimages' ? "720px" : "auto"}
    >
        {#each rows as row, i}
            <Card
                id={i.toString()}
                pivot={{data: row[main_index], cell_component: pivot_cell_component, cell_props: {height: layout === 'gimages' ? `${pivot_height}vh` : ""}}}
                content={tag_indices.map((z) => row[z])}
                layout={layout}
                pivot_tooltip={true}
                content_tooltip={false}
                card_flex_grow={false}
                --card-width=""
            >
                <div slot="pivot-tooltip">
                    <InfoCard
                        pivot={{data: row[main_index], cell_component: pivot_cell_component, cell_props: {height: "100%"}}}
                        pivot_header={main_column}
                        content={row.filter((z, i) => i !== main_index)}
                        content_headers={columns.filter((z, i) => i !== main_index)}
                        card_flex_grow={false}
                        --card-width=""
                    />
                </div>
            </Card>
        {/each}
    </div>
</div>

<style>
	.panel {
		@apply dark:bg-slate-600;
	}

	.panel-masonry {
		/* Solution 1: multiple columns in a masonry layout */
		@apply columns-3;
	}

	.panel-gimages {
		/* Solution 2: flex containers in the Google Images style */
		@apply flex flex-wrap justify-around;
	}
</style>

<script lang="ts">
	import { api_url } from '../../../routes/network/stores';
	import { get_info } from '$lib/api/sliceby';
	import { get_schema } from '$lib/api/datapanel';
	import SliceCard from './SliceCard.svelte';

	export let sliceby_id: string;
	export let datapanel_id: string;
	export let main_column: string;
	export let tag_columns: Array<string>;

	$: schema_promise = get_schema($api_url, datapanel_id);
	$: info_promise = get_info($api_url, sliceby_id);
	//$: rows_promise = get_rows($api_url, datapanel_id, page * per_page, (page + 1) * per_page);
</script>

<div class="isolate bg-white container mx-auto space-y-2 p-2">
	{#await schema_promise}
		<div>Loading schema...</div>
	{:then schema}
		{#await info_promise}
			<div class="h-full">Loading data...</div>
		{:then info}
			<div class="overflow-y-auto overflow-x-hidden h-full">
				{#each info.slice_keys as slice_key}
					<SliceCard {sliceby_id} {datapanel_id} {slice_key} {schema} {main_column} {tag_columns} />
				{/each}
			</div>
		{/await}
	{/await}

	<!-- {#each gallery.slices as slice}
        <Panel name={slice.name} descriptions={slice.descriptions} stats={slice.stats} instances={slice.instances} plot={slice.plot}/>
    {/each} -->
</div>

<script lang="ts">
	import { fetchSchema } from '$lib/utils/api';

	import type { Writable } from 'svelte/store';
	import SliceByCard from './SliceByCard.svelte';

	export let sliceby: Writable<SliceByBox>;
	export let df: Writable<DataFrameBox>;
	export let main_column: Writable<string>;
	export let tag_columns: Writable<Array<string>>;
	export let aggregations: any;

	$: schema_promise = fetchSchema($df);
	$: info_promise = get_sliceby_info($sliceby.ref_id);
	let aggregations_promise = aggregate_sliceby($sliceby.ref_id, (aggregations = aggregations));
</script>

<div class="isolate bg-white container mx-auto space-y-2 p-2">
	{#await schema_promise}
		<div>Loading schema...</div>
	{:then schema}
		{#await info_promise}
			<div class="h-full">Loading data...</div>
		{:then info}
			<div class="flex-rows space-y-3 overflow-y-auto overflow-x-hidden h-full">
				{#each info.slice_keys as slice_key}
					<SliceByCard
						{sliceby}
						{slice_key}
						{schema}
						main_column={$main_column}
						tag_columns={$tag_columns}
						{aggregations_promise}
					/>
				{/each}
			</div>
		{/await}
	{/await}
</div>

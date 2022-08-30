<script lang="ts">
	import { api_url } from '$network/stores.js';
	import Pagination from '$lib/components/pagination/Pagination.svelte';
	import Table from '$lib/components/table/Table.svelte';
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';

	const { get_schema, get_rows, add, match } = getContext('Interface');

	export let dp: Writable;

	export let page: number = 0;
	export let per_page: number = 100;

	$: schema_promise = $get_schema($dp.box_id);
	$: rows_promise = $get_rows($dp.box_id, page * per_page, (page + 1) * per_page);
</script>

<div class="bg-slate-100">
	{$dp.box_id}
	{#await schema_promise}
		waiting....
	{:then schema}
		<div class="table-view">
			<div class="overflow-y-auto overflow-x-hidden h-full">
				{#await rows_promise}
					<Table rows={null} {schema} />
				{:then rows}
					<Table rows={rows} {schema} />
				{:catch error}
					{error}
				{/await}
			</div>

			<div class="z-10 top-0 m-0 h-20">
				<Pagination bind:page bind:per_page loaded_items={schema.nrows} total_items={schema.nrows} />
			</div>
		</div>
	{:catch error}
		{error}
	{/await}
</div>

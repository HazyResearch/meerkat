<script lang="ts">
	import { get_rows,get_schema } from '$lib/api/datapanel';
	import Table from '$lib/components/table/Table.svelte';
	import { api_url } from '../routes/network/stores';
	import Pagination from '$lib/components/pagination/Pagination.svelte';
	import SearchHeader from '$lib/components/search_header/SearchHeader.svelte';

	export let per_page: number = 10;

	export let nrows: number = 0;
	export let datapanel_id: string = '';
	export let page: number = 0;

	$: schema_promise = get_schema($api_url, datapanel_id);
	$: rows_promise = get_rows($api_url, datapanel_id, page * per_page, (page + 1) * per_page);
</script>

<SearchHeader datapanel_id={datapanel_id} schema_promise={schema_promise}></SearchHeader>
<div class="table-view">
	{#await schema_promise}
		<div class="h-full">Loading data...</div>
	{:then schema}
		<div class="overflow-y-auto overflow-x-hidden h-full">
			{#await rows_promise}
				<Table rows={null} {schema} />
			{:then rows}
				<Table {rows} {schema} />
			{/await}
		</div>
	{/await}
	<div class="z-10 top-0 m-0 h-20">
		<Pagination bind:page bind:per_page loaded_items={nrows} total_items={nrows} />
	</div>
</div>

<style>
	.table-view {
		@apply h-full grid grid-rows-[1fr_auto] overflow-hidden;
		@apply dark:bg-slate-700;
	}
</style>

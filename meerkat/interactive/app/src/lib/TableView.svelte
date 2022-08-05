<script lang="ts">
	import Table from '$lib/components/table/Table.svelte';
	import { post } from '$lib/utils/requests';
	import { get_rows, type DataPanelRows } from '$lib/api/datapanel';
	import { api_url } from '../routes/network/stores';
	import Pagination from './components/pagination/Pagination.svelte';

	console.log('in tableview');
	export let per_page: number = 10;

	export let nrows: number = 0;
	export let dp_id: string = '';
	export let page: number = 0;

	$: datapanel_promise = get_rows($api_url, dp_id, page * per_page, (page + 1) * per_page)
</script>

<div class="table-view">
	{#await datapanel_promise}
		<div class="h-full">Loading data...</div>
	{:then datapanel}
		<div class="overflow-y-auto overflow-x-hidden h-full">
			<Table {datapanel} />
		</div>
	{/await}
	<div class="z-10 top-0 m-2 h-20">
		<Pagination bind:page bind:per_page loaded_items={nrows} total_items={nrows} />
	</div>
</div>

<style>
	.table-view {
		@apply h-full grid grid-rows-[1fr_auto] overflow-hidden;
		@apply dark:bg-slate-700;
	}
</style>

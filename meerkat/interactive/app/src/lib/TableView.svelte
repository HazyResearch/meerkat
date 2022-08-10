<script lang="ts">
	import { get_rows, get_schema } from '$lib/api/datapanel';
	import type { RefreshCallback } from '$lib/api/callbacks';
	import Table from '$lib/components/table/Table.svelte';
	import { api_url } from '../routes/network/stores';
	import Pagination from '$lib/components/pagination/Pagination.svelte';
	import SearchHeader from '$lib/components/search_header/SearchHeader.svelte';
	import Toggle from './components/common/Toggle.svelte';
	import Gallery from './components/gallery/Gallery.svelte';

	export let datapanel_id: string;
	export let nrows: number = 0;
	export let page: number = 0;
	export let per_page: number = 10;

	const base_datapanel_id: string = datapanel_id;

	$: schema_promise = get_schema($api_url, datapanel_id);
	$: rows_promise = get_rows($api_url, datapanel_id, page * per_page, (page + 1) * per_page);

	let toggle_button: boolean = false;
	$: active_view = toggle_button ? 'gallery' : 'table';

	const refresh: RefreshCallback = (new_datapanel_id: string) => {
		datapanel_id = new_datapanel_id;
	}

</script>

<div class="inline-flex mb-4">
	<Toggle label_left="Table" label_right="Gallery" bind:checked={toggle_button} />
</div>

<SearchHeader base_datapanel_id={base_datapanel_id} schema_promise={schema_promise} refresh_callback={refresh}></SearchHeader>
{#if active_view === 'table'}
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
{:else if active_view === 'gallery'}
	<div class="table-view">
		{#await schema_promise}
			<div class="h-full">Loading gallery...</div>
		{:then schema}
			<div class="overflow-y-auto overflow-x-hidden h-full">
				{#await rows_promise}
					<Gallery
						{schema}
						rows={{rows: []}}
					/>
				{:then rows}
					<Gallery
						{schema}
						{rows}
						main_column={'img'}
						tag_columns={['label_idx', 'split']}
					/>
				{/await}
			</div>
		{/await}
		<div class="z-10 top-0 m-0 h-20">
			<Pagination bind:page bind:per_page loaded_items={nrows} total_items={nrows} />
		</div>
	</div>
{/if}

<style>
	.table-view {
		@apply h-full grid grid-rows-[1fr_auto] overflow-hidden;
		@apply dark:bg-slate-700;
	}
</style>

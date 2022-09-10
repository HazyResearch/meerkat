<script lang="ts">
	import { api_url } from '$network/stores.js';
	import Pagination from '$lib/components/pagination/Pagination.svelte';
	import Gallery from '$lib/components/gallery/Gallery.svelte';
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { get } from 'svelte/store';
	import { createEventDispatcher } from 'svelte';
	import type { ColumnInfo } from '$lib/api/datapanel.ts';
	import { BarLoader } from 'svelte-loading-spinners'


	const { get_schema, get_rows, edit } = getContext('Interface');

	export let dp: Writable;
	export let main_column: Writable<string>;
	export let tag_columns: Writable<Array<string>>;
	export let edit_target: Any;

	export let page: number = 0;
	export let per_page: number = 100;

	$: schema_promise = $get_schema($dp.box_id);
	$: rows_promise = $get_rows($dp.box_id, page * per_page, (page + 1) * per_page);

	async function handle_edit(event: any) {
		let { pivot, pivot_id_column, id_column } = edit_target;
		let rows = await rows_promise;

		let { row, column, value } = event.detail;
		let row_id_column_index = rows.column_infos.findIndex((c) => c.name === id_column);
		let row_index = rows.indices.indexOf(row);
		let row_id = rows.rows[row_index][row_id_column_index];

		$edit(pivot.box_id, value, column, row_id, pivot_id_column);
	}
</script>

<div class="flex-1">
	{#await schema_promise}
		<div class="flex justify-center items-center h-full">
			<BarLoader size="80" color="#7c3aed" unit="px" duration="1s"></BarLoader>
		</div>
	{:then schema}
		<div class="h-full flex flex-col">
			<div class="grow overflow-y-scroll">
				{#await rows_promise}
				<div class="justify-center items-center">
					<BarLoader size="80" color="#7c3aed" unit="px" duration="1s"></BarLoader>
				</div>
				{:then rows}
					<Gallery {schema} {rows} main_column={$main_column} tag_columns={$tag_columns} />
				{:catch error}
					{error}
				{/await}
			</div>

			<div class="z-10 top-0 m-0 h-20 px-10">
				<Pagination
					bind:page
					bind:per_page
					loaded_items={schema.nrows}
					total_items={schema.nrows}
				/>
			</div>
		</div>
	{:catch error}
		{error}
	{/await}
</div>

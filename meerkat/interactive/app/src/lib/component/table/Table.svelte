<script lang="ts">
	import type { DataFrameChunk } from '$lib/api/dataframe';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import Table from '$lib/shared/table/Table.svelte';
	import { createEventDispatcher, getContext } from 'svelte';

	const dispatch = createEventDispatcher();

	const { get_schema, fetch_chunk } = getContext('Meerkat');

	export let df;

	export let page: number = 0;
	export let per_page: number = 100;
	export let editable: boolean = false;
	export let id_column: string = null;

	export let column_widths: Array<number>;

	$: schema_promise = get_schema(df.ref_id);
	$: rows_promise = fetch_chunk({ df: df, start: page * per_page, end: (page + 1) * per_page });

	$: schema_promise.then((s: any) => {
		if (column_widths == null) {
			column_widths = Array.apply(null, Array(s.columns.length)).map((x, i) => 256);
		}
	});

	async function handle_edit(event: any) {
		let rows: DataFrameChunk = await rows_promise;

		let { row, column, value } = event.detail;
		let row_id_column_index = rows.column_infos.findIndex((c) => c.name === id_column);
		let row_index = rows.indices.indexOf(row);
		let row_id = rows.rows[row_index][row_id_column_index];

		dispatch('edit', {
			row: row,
			row_id: row_id,
			column: column,
			value: event.detail.value
		});
	}
</script>

<div class="h-full flex-1 rounded-lg overflow-hidden bg-slate-50">
	{#await schema_promise}
		waiting....
	{:then schema}
		<div class="h-full">
			<div class="h-full overflow-y-scroll pb-28">
				{#await rows_promise}
					<Table rows={null} {schema} {column_widths} />
				{:then rows}
					<Table {rows} {schema} {column_widths} {editable} {id_column} on:edit={handle_edit} />
				{:catch error}
					{error}
				{/await}
			</div>
			<div class="absolute z-10 bottom-0 w-[90%] left-[5%] mb-8">
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

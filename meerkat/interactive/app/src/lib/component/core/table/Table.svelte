<script lang="ts">
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import Table from '$lib/shared/table/Table.svelte';
	import { fetchChunk, fetchSchema } from '$lib/utils/api';
	import type { DataFrameChunk, DataFrameRef } from '$lib/utils/dataframe';
	import { createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher();

	export let df: DataFrameRef;

	export let page: number = 0;
	export let perPage: number = 100;
	export let editable: boolean = false;
	export let idColumn: string = null;

	export let columnWidths: Array<number>;

	$: schemaPromise = fetchSchema({ df: df });
	$: rowsPromise = fetchChunk({ df: df, start: page * perPage, end: (page + 1) * perPage });

	$: schemaPromise.then((s: any) => {
		if (columnWidths == null) {
			columnWidths = Array.apply(null, Array(s.columns.length)).map((x, i) => 256);
		}
	});

	async function handle_edit(event: any) {
		let rows: DataFrameChunk = await rowsPromise;

		let { row, column, value } = event.detail;
		let rowIdColumnIndex = rows.columnInfos.findIndex((c) => c.name === idColumn);
		let rowIndex = rows.indices.indexOf(row);
		let rowId = rows.rows[rowIndex][rowIdColumnIndex];

		dispatch('edit', {
			row: row,
			row_id: rowId,
			column: column,
			value: event.detail.value
		});
	}
</script>

<div class="h-full flex-1 rounded-lg overflow-hidden bg-slate-50">
	{#await schemaPromise}
		waiting....
	{:then schema}
		<div class="relative h-full">
			<div class="h-full overflow-y-scroll pb-28">
				{#await rowsPromise}
					<Table rows={null} {schema} {columnWidths} />
				{:then rows}
					<Table {rows} {schema} {columnWidths} {editable} {idColumn} on:edit={handle_edit} />
				{:catch error}
					{error}
				{/await}
			</div>
			<div class="absolute z-10 bottom-0 w-[90%] left-[5%] mb-8">
				<Pagination bind:page bind:perPage totalItems={schema.nrows} />
			</div>
		</div>
	{:catch error}
		{error}
	{/await}
</div>

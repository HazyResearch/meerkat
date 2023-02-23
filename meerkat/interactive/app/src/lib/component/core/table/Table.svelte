<script lang="ts">
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema, dispatch } from '$lib/utils/api';
	import type { DataFrameChunk, DataFrameRef, DataFrameSchema } from '$lib/utils/dataframe';
	import { Dropdown, DropdownItem } from 'flowbite-svelte';
	import { setContext, getContext } from 'svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import { openModal } from 'svelte-modals';
	import type { Endpoint } from '$lib/utils/types';
	import { get, readable, writable } from 'svelte/store';

	import { Render, Subscribe, createTable, createRender } from 'svelte-headless-table';
	import { addResizedColumns } from 'svelte-headless-table/plugins/';
	import Text from '../text/Text.svelte';
	import Cell from '$lib/shared/cell/Cell.svelte';

	export let df: DataFrameRef;
	export let selected: Array<string>;

	export let page: number = 0;
	export let perPage: number = 20;

	export let allowSelection: boolean = false;
	export let onEdit: Endpoint | null = null;

	let schemaPromise: Promise<DataFrameSchema>;
	$: schemaPromise = fetchSchema({
		df: df,
		formatter: 'small'
	});

	setContext('open_row_modal', (posidx: number) => {
		openModal(RowModal, {
			df: df,
			posidx: posidx,
			mainColumn: undefined
		});
	});

	const data = writable([]);

	$: chunkPromise = fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		formatter: 'tiny'
	});

	const buildTable = (columns: any) => {
		const table = createTable(data, {
			resize: addResizedColumns()
		});
		columns = table.createColumns(
			columns.map((column: any) => {
				return table.column({
					header: column.name,
					accessor: column.name,
					plugins: {
						resize: {}
					},
					cell: (item) =>
						createRender(Cell, {
							data: item.value,
							cellComponent: column.cellComponent,
							cellProps: column.cellProps,
							cellDataProp: column.cellDataProp,
							editable: true
						}).on('edit', (e) => {
							dispatch(onEdit.endpointId, {
								detail: {
									column: item.id,
									keyidx: item.row.dataId,
									value: e.detail.value
								}
							});
						})
				});
			})
		);
		return { table: table, columns: columns };
	};
	const buildInitialTable = async (schema: DataFrameSchema) => {
		const { table, columns } = buildTable(schema.columns);
		return table.createViewModel(columns);
	};
	const buildFullTable = async (chunk: DataFrameChunk) => {
		data.set(chunk.getRows());
		const { table, columns } = buildTable(chunk.columnInfos);
		return table.createViewModel(columns, {
			rowDataId: (item, index) => {
				return item[chunk.primaryKey];
			}
		});
	};
	$: initialModelPromise = schemaPromise.then(buildInitialTable);

	$: fullModelPromise = chunkPromise.then(buildFullTable);
</script>


{#await schemaPromise then schema}
	<div class="bg-slate-50 w-fit rounded-md py-1">
		{#await initialModelPromise then { headerRows, tableAttrs }}
			<table {...get(tableAttrs)} class="bg-white border-solid border-spacing-0">
				<thead class="bg-slate-50 text-slate-800">
					{#each get(headerRows) as headerRow (headerRow.id)}
						<Subscribe
							rowAttrs={headerRow.attrs()}
							let:rowAttrs
							rowProps={headerRow.props()}
							let:rowProps
						>
							<tr {...rowAttrs}>
								{#each headerRow.cells as cell (cell.id)}
									<Subscribe attrs={cell.attrs()} props={cell.props()} let:attrs let:props>
										<th {...attrs} use:props.resize class="text-left font-mono">
											<div class="resizer bg-slate-50" use:props.resize.drag />
											<Render of={cell.render()} />
										</th>
									</Subscribe>
								{/each}
							</tr>
						</Subscribe>
					{/each}
				</thead>
				{#await fullModelPromise then { rows, tableBodyAttrs }}
					<tbody {...get(tableBodyAttrs)}>
						{#each get(rows) as row (row.id)}
							<Subscribe rowAttrs={row.attrs()} let:rowAttrs>
								<tr {...rowAttrs}>
									{#each row.cells as cell (cell.id)}
										<Subscribe attrs={cell.attrs()} let:attrs>
											<td
												{...attrs}
												class="border-slate-200 border-2 border-solid border-spacing-0"
											>
												<Render of={cell.render()} />
											</td>
										</Subscribe>
									{/each}
								</tr>
							</Subscribe>
						{/each}
					</tbody>
				{/await}
			</table>
		{/await}
		<div class="flex self-center justify-self-end items-center">
			<Pagination bind:page bind:perPage totalItems={schema.nrows} />
		</div>
	</div>
{/await}

<style>
	th {
		position: relative;
	}

	.resizer {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 2px;
		right: -1;
		cursor: col-resize;
		z-index: 1;
	}
</style>

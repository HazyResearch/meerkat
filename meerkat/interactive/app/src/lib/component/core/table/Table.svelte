<script lang="ts">
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema } from '$lib/utils/api';
	import type { DataFrameChunk, DataFrameRef, DataFrameSchema } from '$lib/utils/dataframe';
	import { Dropdown, DropdownItem } from 'flowbite-svelte';
	import { setContext, getContext } from 'svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import { openModal } from 'svelte-modals';
	import { get, readable, writable } from 'svelte/store';

	import { Render, Subscribe, createTable, createRender } from 'svelte-headless-table';
	import { addResizedColumns } from 'svelte-headless-table/plugins/';
	import Text from '../text/Text.svelte';
	import Cell from '$lib/shared/cell/Cell.svelte';

	export let df: DataFrameRef;
	export let selected: Array<string>;

	export let page: number = 0;
	export let perPage: number = 20;
	export let cellSize: number = 24;

	export let allowSelection: boolean = false;

	const components: { [key: string]: ComponentType } = getContext('Components');
	const componentId = getContext('componentId');

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

	$: chunkPromise = fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		formatter: 'tiny'
	});

	let dropdownOpen: boolean = false;

	const data = writable([]);

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
					cell: ({ value, id }) =>
						createRender(Cell, {
							data: value,
							cellComponent: column.cellComponent,
							cellProps: column.cellProps,
							cellDataProp: column.cellDataProp,
							editable: true
						}).on("edit", () => {console.log("editterr", id, cellId)})
				});
			})
		);
		return table.createViewModel(columns);
	};
	const buildInitialTable = async (schema: DataFrameSchema) => {
		return buildTable(schema.columns);
	};
	$: initialModelPromise = schemaPromise.then(buildInitialTable);
	const buildFullTable = async (chunk: DataFrameChunk) => {
		$data = chunk.getRows();
		return buildTable(chunk.columnInfos);
	};
	$: fullModelPromise = chunkPromise.then(buildFullTable);
</script>

<div class="">
	{#await schemaPromise}
		pass
	{:then schema}
		<div class="flex self-center justify-self-end items-center">
			<Pagination bind:page bind:perPage totalItems={schema.nrows} />
		</div>
		{#await initialModelPromise then { headerRows, rows, tableAttrs, tableBodyAttrs }}
			<table {...get(tableAttrs)}>
				<thead>
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
										<th {...attrs} use:props.resize>
											<Render of={cell.render()} />
											<div class="resizer" use:props.resize.drag />
										</th>
									</Subscribe>
								{/each}
							</tr>
						</Subscribe>
					{/each}
				</thead>
				{#await fullModelPromise then { headerRows, rows, tableAttrs, tableBodyAttrs }}
					<tbody {...get(tableBodyAttrs)}>
						{#each get(rows) as row (row.id)}
							<Subscribe rowAttrs={row.attrs()} let:rowAttrs>
								<tr {...rowAttrs}>
									{#each row.cells as cell (cell.id)}
										<Subscribe attrs={cell.attrs()} let:attrs>
											<td {...attrs}>
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
	{/await}
</div>

<style>
	th {
		position: relative;
	}

	.resizer {
		position: absolute;
		top: 0;
		bottom: 0;
		right: -4px;
		width: 8px;
		background: lightgray;
		cursor: col-resize;
		z-index: 1;
	}
</style>

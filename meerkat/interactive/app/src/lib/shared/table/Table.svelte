<script lang="ts">
	import Cell from '$lib/shared/cell/Cell.svelte';
	import type { ColumnInfo, DataFrameChunk, DataFrameSchema } from '$lib/utils/dataframe';
	import { createEventDispatcher, onMount } from 'svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import { zip } from 'underscore';

	const dispatch = createEventDispatcher();

	export let rows: DataFrameChunk | null;
	export let schema: DataFrameSchema;
	export let editable: boolean = false;
	let columnInfos: Array<ColumnInfo> = schema.columns;

	export let columnWidths = Array.apply(null, Array(columnInfos.length)).map((x, i) => 256);
	let columnUnit: string = 'px';

	let resizeProps = {
		colBeingResized: -1,
		x: 0,
		wLeft: 0,
		wRight: 0,
		dx: 0
	};

	let resizeMethods = {
		mousedown(colIndex: number) {
			return (e: MouseEvent) => {
				// Update all the resize props
				resizeProps.colBeingResized = colIndex;
				resizeProps.x = e.clientX;
				resizeProps.wLeft = columnWidths[colIndex];
				resizeProps.wRight = columnWidths[colIndex + 1];

				// Attach listeners for events
				window.addEventListener('mousemove', resizeMethods.mousemove);
				window.addEventListener('mouseup', resizeMethods.mouseup);
			};
		},

		mousemove(e: MouseEvent) {
			if (resizeProps.colBeingResized === -1) return;

			// Determine how far the mouse has been moved
			resizeProps.dx = e.clientX - resizeProps.x;

			// Update the width of column
			if (resizeProps.wLeft + resizeProps.dx > 164 && resizeProps.wRight - resizeProps.dx > 164) {
				columnWidths[resizeProps.colBeingResized] = resizeProps.wLeft + resizeProps.dx;
				columnWidths[resizeProps.colBeingResized + 1] = resizeProps.wRight - resizeProps.dx;
			}
		},

		mouseup(e: MouseEvent) {
			window.removeEventListener('mousemove', resizeMethods.mousemove);
			window.removeEventListener('mouseup', resizeMethods.mouseup);
		}
	};

	let tableWidth: number;
	onMount(async () => {
		columnWidths = Array.apply(null, Array(columnInfos.length)).map(
			(x, i) => tableWidth / columnInfos.length
		);
		columnUnit = 'px';
	});

	function handleEdit(event: any, row: number, column: string) {
		dispatch('edit', {
			row: row,
			column: column,
			value: event.detail.value
		});
	}
</script>

<div
	class="table pl-4 pr-4 table-fixed overflow-x-scroll text-sm w-full h-full bg-slate-200"
>
	<div class="table-header-group">
		<div class="table-row" bind:clientWidth={tableWidth}>
			{#each columnInfos as column, col_index}
				<div class="table-cell" style="width:{columnWidths[col_index]}{columnUnit}">
					<slot id="header-cell">
						<div class="flex flex-col items-center">
							<div class="pb-1 font-bold">{column.name}</div>
							<div
								class="text-clip bg-violet-200 font-mono text-xs text-slate-500 rounded-full px-3 py-0.5"
							>
								{column.type}
							</div>
						</div>
					</slot>
					<div class="resizer rounded-md" on:mousedown={resizeMethods.mousedown(col_index)} />
				</div>
			{/each}
		</div>
	</div>

	<div class="table-row-group">
		{#if rows}
			{#each zip(rows.rows, rows.indices) as [row, index]}
				<div class="table-row">
					{#each zip(row, rows.columnInfos) as [value, columnInfo]}
						<div class="table-cell align-middle p-5">
							<Cell
								data={value}
								cellComponent={columnInfo.cellComponent}
								cellProps={columnInfo.cellProps}
								cellDataProp={columnInfo.cellDataProp}
								editable={editable && columnInfo.name !== idColumn}
								on:edit={(event) => handleEdit(event, index, columnInfo.name)}
							/>
						</div>
					{/each}
				</div>
			{/each}
		{/if}
	</div>
</div>
{#if !rows}
	<div class="flex justify-center items-center h-full">
		<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
	</div>
{/if}

<style>
	.table-header-group .table-cell {
		@apply sticky top-0 py-2 px-4; /* sticky-top */
		/* @apply resize-x [overflow:hidden]; resizing */
		@apply dark:bg-gray-700 dark:text-slate-400;
		@apply bg-slate-100 drop-shadow-xl rounded-l;
	}

	.table-header-group .table-cell:not(:last-child) .resizer {
		@apply absolute h-60 right-0 top-0 w-2 cursor-col-resize;
		user-select: none;
	}

	.table-header-group .table-cell:not(:last-child) .resizer:hover {
		@apply absolute h-full right-0 top-0 w-2 cursor-col-resize;
		@apply bg-no-repeat bg-center bg-[length:2px_100%] bg-gradient-to-b from-slate-100 to-slate-100;
	}

	.table-row-group .table-row {
		@apply border-b border-slate-100 dark:bg-gray-800 dark:border-gray-700  dark:hover:bg-gray-600;
		@apply h-12 overflow-y-scroll;
	}

	.table-row-group .table-cell {
		@apply border-b border-slate-200 px-4 text-left break-words;
	}
</style>

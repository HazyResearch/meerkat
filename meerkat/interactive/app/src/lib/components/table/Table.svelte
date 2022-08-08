<script lang="ts">
	import type { DataPanelRows, ColumnInfo, DataPanelSchema } from '$lib/api/datapanel';
	import Cell from '$lib/components/cell/Cell.svelte';
	import { onMount } from 'svelte';
	import { zip } from 'underscore';
	import { BarLoader } from 'svelte-loading-spinners'


	export let rows: DataPanelRows | null;
	export let schema: DataPanelSchema;
	let column_infos: Array<ColumnInfo> = schema.columns; 

	let column_widths = Array.apply(null, Array(column_infos.length)).map((x, i) => 100 / column_infos.length);
	let column_unit: string = '%';

	let resize_props = {
		col_being_resized: -1,
		x: 0,
		w_left: 0,
		w_right: 0,
		dx: 0
	};

	let resize_methods = {
		mousedown(col_index: number) {
			return (e: MouseEvent) => {
				// Update all the resize props
				resize_props.col_being_resized = col_index;
				resize_props.x = e.clientX;
				resize_props.w_left = column_widths[col_index];
				resize_props.w_right = column_widths[col_index + 1];

				// Attach listeners for events
				window.addEventListener('mousemove', resize_methods.mousemove);
				window.addEventListener('mouseup', resize_methods.mouseup);
			};
		},

		mousemove(e: MouseEvent) {
			if (resize_props.col_being_resized === -1) return;

			// Determine how far the mouse has been moved
			resize_props.dx = e.clientX - resize_props.x;

			// Update the width of column
			if (
				resize_props.w_left + resize_props.dx > 40 &&
				resize_props.w_right - resize_props.dx > 40
			) {
				column_widths[resize_props.col_being_resized] = resize_props.w_left + resize_props.dx;
				column_widths[resize_props.col_being_resized + 1] = resize_props.w_right - resize_props.dx;
			}
		},

		mouseup(e: MouseEvent) {
			window.removeEventListener('mousemove', resize_methods.mousemove);
			window.removeEventListener('mouseup', resize_methods.mouseup);
		}
	};

	let table_width: number;
	onMount(async () => {
		column_widths = Array.apply(null, Array(column_infos.length)).map(
			(x, i) => table_width / column_infos.length
		);
		column_unit = 'px';
	});
</script>

<div class="table">
	<div class="table-header-group">
		<div class="table-row" bind:clientWidth={table_width}>
			{#each column_infos as column, col_index}
				<div class="table-cell" style="width:{column_widths[col_index]}{column_unit}">
					<slot id="header-cell">
						<div class="flex flex-col items-center">
							<div class="pb-1 font-bold">{column.name}</div>
							<div class="text-clip bg-violet-200  font-mono text-slate-500 rounded-full px-3 py-0.5">
								{column.type}
							</div>
						</div>
					</slot>
					<div class="resizer rounded-md" on:mousedown={resize_methods.mousedown(col_index)} />
				</div>
			{/each}
		</div>
	</div>

	<div class="table-row-group bg-white">
		{#if rows}
			{#each rows.rows as row}
				<div class="table-row">
					{#each zip(row, rows.column_infos) as [value, column_info]}
						<div class="table-cell align-middle p-5">
							<Cell data={value} cell_component={column_info.cell_component} cell_props={column_info.cell_props}/>
						</div>
					{/each}
				</div>
			{/each}
		{/if}
	</div>
</div>
{#if !rows}
<div class="flex justify-center items-center h-full">
	<BarLoader size="80" color="#7c3aed" unit="px" duration="1s"></BarLoader>
</div>
{/if}

<style>
	.table {
		@apply pl-4 pr-4 table-fixed text-sm w-full;
		@apply dark:text-gray-300 dark:bg-gray-700;
	}

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
	}

	.table-row-group .table-cell {
		@apply border-b border-slate-200 px-4 text-left overflow-hidden break-words;
	}
</style>

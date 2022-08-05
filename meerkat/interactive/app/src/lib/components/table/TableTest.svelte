<script lang="ts">
	import type { DataPanelRows, ColumnInfo } from '$lib/api/datapanel';
	import Cell from '$lib/components/cell/Cell.svelte';
	import { onMount } from 'svelte';
	import { zip } from 'underscore';

	export let datapanel: DataPanelRows;
	let column_infos: Array<ColumnInfo> = datapanel.column_infos; 

	// let column_widths = Array.apply(null, Array(column_infos.length)).map((x, i) => 100 / column_infos.length);
	// let column_unit: string = '%';

</script>

<div class="table">
	{column_infos}
</div>

<style>
	.table {
		@apply pl-4 pr-4 table-fixed text-xs w-full;
		@apply dark:text-gray-300 dark:bg-gray-700;
	}

	.table-header-group .table-cell {
		@apply sticky top-0 py-2 px-4; /* sticky-top */
		/* @apply resize-x [overflow:hidden]; resizing */
		@apply dark:bg-gray-700 dark:text-slate-400;
		@apply uppercase;
	}

	.table-header-group .table-cell:not(:last-child) .resizer {
		@apply absolute h-full right-0 top-0 w-2 cursor-col-resize;
		user-select: none;
	}

	.table-header-group .table-cell:not(:last-child) .resizer:hover {
		@apply absolute h-full right-0 top-0 w-2 cursor-col-resize;
		@apply bg-no-repeat bg-center bg-[length:2px_100%] bg-gradient-to-b from-slate-100 to-slate-100;
	}

	.table-row-group .table-row {
		@apply dark:bg-gray-800 dark:border-gray-700  dark:hover:bg-gray-600;
	}

	.table-row-group .table-cell {
		@apply px-4 text-left overflow-hidden break-words;
	}
</style>

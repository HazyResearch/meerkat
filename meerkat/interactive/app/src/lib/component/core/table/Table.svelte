<script lang="ts">
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema, dispatch } from '$lib/utils/api';
	import {
		DataFrameChunk,
		type ColumnInfo,
		type DataFrameRef,
		type DataFrameSchema
	} from '$lib/utils/dataframe';
	import { setContext, getContext } from 'svelte';
	import { createEventDispatcher, onMount } from 'svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import { openModal } from 'svelte-modals';
	import type { Endpoint } from '$lib/utils/types';
	import { zip } from 'underscore';
	import { get, readable, writable, type Writable } from 'svelte/store';
	import Cell from '$lib/shared/cell/Cell.svelte';

	export let df: DataFrameRef;
	export let selected: Array<string>;

	export let page: number = 0;
	export let perPage: number = 20;

	export let allowSelection: boolean = false;
	export let onEdit: Endpoint;

	// Setup row modal
	setContext('open_row_modal', (posidx: number) => {
		openModal(RowModal, {
			df: df,
			posidx: posidx,
			mainColumn: undefined
		});
	});
	console.log('here');

	// Create placeholder variables for table data.
	let schema: Writable<DataFrameSchema> = writable({
		columns: [],
		primaryKey: 'pkey',
		nrows: 0,
		id: ''
	});
	let chunk: Writable<DataFrameChunk> = writable(new DataFrameChunk([], [], [], 0, 'pkey'));

	$: fetchSchema({
		df: df,
		formatter: 'icon'
	}).then((newSchema) => {
		schema.set(newSchema);
	});

	$: fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		formatter: 'tiny'
	}).then((newChunk) => {
		chunk.set(newChunk);
	});

	export let columnWidths = Array.apply(null, Array($schema.columns.length)).map((x, i) => 256);
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
		columnWidths = Array.apply(null, Array($schema.columns.length)).map(
			(x, i) => tableWidth / $schema.columns.length
		);
		columnUnit = 'px';
	});

	$: {
		console.log($chunk);
		console.log($schema);
	}
</script>

<div class="w-fit">
	<div class="auto-table table-fixed overflow-x-scroll text-sm border-collapse">
		<div class="table-header-group">
			<div class="table-row bg-slate-100">
				<!-- bind:clientWidth={tableWidth} -->
				<div class="table-cell border border-slate-300 font-mono text-slate-800 " />
				{#each $schema.columns as column, col_index}
					<div
						class="table-cell border border-slate-300 font-mono text-slate-800 pl-1"
						style="width:{columnWidths[col_index]}{columnUnit}"
					>
						<slot id="header-cell">
							<div class="flex items-center">
								<Cell
									data={''}
									cellComponent={column.cellComponent}
									cellProps={column.cellProps}
									cellDataProp={column.cellDataProp}
								/>
								<div class="">{column.name}</div>
							</div>
						</slot>
						<div class="resizer rounded-md" on:mousedown={resizeMethods.mousedown(col_index)} />
					</div>
				{/each}
			</div>
		</div>

		<div class="table-row-group">
			{#each zip($chunk.keyidxs, $chunk.posidxs) as [keyidx, posidx], rowi}
				<div class="table-row items-center">
					<div class="table-cell border border-slate-300 font-mono text-slate-800 bg-slate-100">
						<div class="w-7 text-center">
							{posidx}
						</div>
					</div>
					{#each $schema.columns as col}
						<div class="table-cell border border-slate-200">
							<Cell
								{...$chunk.getCell(rowi, col.name)}
								editable={true}
								on:edit={(e) => {
									dispatch(onEdit.endpointId, {
										detail: {
											column: col.name,
											keyidx: keyidx,
											posidx: posidx,
											value: e.detail.value
										}
									});
								}}
							/>
						</div>
					{/each}
				</div>
			{/each}
		</div>
	</div>
	<div class="grid grid-cols-3 h-8 z-10 bg-slate-100 px-5 rounded-b-sm">
		<!-- Left header section -->
		<div class="flex justify-self-start items-center">
			<!-- <span class="font-semibold">
				<GallerySlider bind:size={cellSize} />
			</span>
			<div class="font-semibold self-center px-10 flex space-x-2">
				{#if selected.length > 0}
					<Selected />
					<div class="text-violet-600">{selected.length}</div>
				{/if}
			</div> -->
		</div>

		<!-- Middle header section -->
		<div class="self-center justify-self-center">
			<!-- <button
				class="font-bold font-mono text-xl text-slate-600 self-center justify-self-center"
				on:click={() => {
					dropdownOpen = !dropdownOpen;
				}}
			>
				{mainColumn}
			</button>
			<Dropdown open={dropdownOpen} class="w-fit">
				{#each schema.columns as col}
					<DropdownItem
						on:click={() => {
							mainColumn = col.name;
							dropdownOpen = false;
						}}
					>
						<div class="text-slate-600 font-mono">
							<span class="font-bold">{col.name}</span>
						</div>
					</DropdownItem>
				{/each}
			</Dropdown> -->
		</div>

		<!-- Right header section -->
		<div class="flex self-center justify-self-end items-center">
			<Pagination bind:page bind:perPage totalItems={$schema.nrows} />
		</div>
	</div>
</div>

<!-- {#if !rows}
	<div class="flex justify-center items-center h-full">
		<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
	</div>
{/if} -->
<style>
	.table-header-group .table-cell {
		@apply sticky top-0; /* sticky-top */
		@apply resize-x [overflow:hidden];
	}

	.table-row-group .table-row {
		@apply overflow-y-scroll;
	}

	.table-row-group .table-cell {
		@apply text-left;
	}
</style>

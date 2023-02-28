<script lang="ts">
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema, dispatch } from '$lib/utils/api';
	import { DataFrameChunk, type DataFrameRef, type DataFrameSchema } from '$lib/utils/dataframe';
	import { without } from 'underscore';
	import { onMount } from 'svelte';
	import { openModal } from 'svelte-modals';
	import type { Endpoint } from '$lib/utils/types';
	import { zip } from 'underscore';
	import { writable, type Writable } from 'svelte/store';
	import Cell from '$lib/shared/cell/Cell.svelte';
	import { Check, CheckAll, KeyFill } from 'svelte-bootstrap-icons';

	export let df: DataFrameRef;

	export let page: number = 0;
	export let perPage: number = 50;

	export let selected: Array<string> = [];
	export let singleSelect: boolean = false;
	export let onEdit: Endpoint;
	export let onSelect: Endpoint;

	export let classes: string = 'h-fit';

	// Setup row modal
	const open_row_modal = (posidx: number) => {
		openModal(RowModal, {
			df: df,
			posidx: posidx,
			mainColumn: undefined
		});
	};

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
		console.log('here');
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
</script>

<!-- FIXME: Figure out how to do h-full -->
<div
	class={'flex-1 w-full bg-slate-100 grid grid-rows-[1fr_auto] rounded-b-md overflow-hidden border-slate-300 ' +
		classes}
>
	<div class="auto-table table-fixed overflow-x-scroll text-sm h-full">
		<div class="table-header-group">
			<div class="table-row sticky top-0 bg-slate-100">
				<!-- bind:clientWidth={tableWidth} -->
				<div class="table-cell border border-slate-300 font-mono text-slate-800" />
				{#each $schema.columns as column, col_index}
					<div
						class="table-cell border border-slate-300 font-mono text-slate-800 pl-1"
						style="width:{columnWidths[col_index]}{columnUnit}"
					>
						<slot id="header-cell">
							<div class="flex items-center gap-1 px-0.5">
								{#if column.name === $schema.primaryKey}
									<!-- Show a key icon for the primary key-->
									<KeyFill class="text-violet-600" />
								{:else}
									<Cell
										data={''}
										cellComponent={column.cellComponent}
										cellProps={column.cellProps}
										cellDataProp={column.cellDataProp}
									/>
								{/if}

								<div class="">{column.name}</div>
							</div>
						</slot>
						<div class="resizer rounded-md" on:mousedown={resizeMethods.mousedown(col_index)} />
					</div>
				{/each}
			</div>
		</div>

		<div class="table-row-group border-collapse">
			{#each zip($chunk.keyidxs, $chunk.posidxs) as [keyidx, posidx], rowi}
				<div class="table-row items-center">
					<div class="table-cell border border-slate-300 font-mono text-slate-800 bg-slate-100">
						<button
							class="w-7 text-center"
							class:text-violet-600={selected.includes(keyidx)}
							class:bg-slate-200={selected.includes(keyidx)}
							on:dblclick={(e) => {
								open_row_modal(posidx);
							}}
							on:click={(e) => {
								let dispatchSelect = true;
								if (e.shiftKey) {
									if (selected.length === 0) {
										selected.push(keyidx);
										dispatchSelect = false;
									} else {
										let lastIdx = selected[selected.length - 1];
										let lasti = $chunk.keyidxs.indexOf(lastIdx);
										let i = $chunk.keyidxs.indexOf(keyidx);
										if (i > lasti) {
											for (let j = lasti; j <= i; j++) {
												if (!selected.includes($chunk.keyidxs[j])) {
													selected.push($chunk.keyidxs[j]);
												}
											}
										} else {
											for (let j = lasti; j >= i; j--) {
												if (!selected.includes($chunk.keyidxs[j])) {
													selected.push($chunk.keyidxs[j]);
												}
											}
										}
									}
								} else if (e.altKey) {
									selected = [];
									selected.push(keyidx);
								} else {
									if (selected.includes(keyidx)) {
										selected = without(selected, keyidx);
									} else if (!selected.includes(keyidx)) {
										if (singleSelect) {
											selected.pop();
										}
										selected.push(keyidx);
									}
								}
								selected = selected;
								if (dispatchSelect) {
									console.log('dispatching onSelect', selected);
									dispatch(onSelect.endpointId, { detail: { selected: selected } });
								}
							}}
						>
							{posidx}
						</button>
					</div>
					{#each $chunk.columnInfos as col}
						<div class="table-cell border-t border-l border-slate-200 hover:opacity-80 bg-white px-1">
							<Cell
								{...$chunk.getCell(rowi, col.name)}
								editable={true}
								on:edit={(e) => {
									console.log(keyidx);
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
	<div class="grid grid-cols-3 h-8 z-10 bg-slate-100 px-5 rounded-b-sm border-t border-t-slate-300">
		<!-- Left header section -->
		<div class="flex justify-self-start items-center">
			<div class="self-center px-2 flex space-x-1 items-center">
				{#if selected.length > 0}
					{#if selected.length === 1}
						<Check class="text-violet-600" />
					{:else}
						<CheckAll class="text-violet-600" />
					{/if}
					<div class="text-violet-600 font-mono text-sm ">{selected.length} Selected</div>
				{/if}
			</div>
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
			<Pagination bind:page bind:perPage totalItems={$schema.nrows} dropdownPlacement={'top'} />
		</div>
	</div>
</div>

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

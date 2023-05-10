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
	let wrap: string = 'clip'; // 'wrap' | 'clip' (add 'overflow' later)

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
		if (columnWidths.length === 0) {
			columnWidths = Array($schema.columns.length).fill(100);
		}
	});

	$: fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		formatter: 'tiny'
	}).then((newChunk) => {
		console.log('here');
		chunk.set(newChunk);
		if (rowHeights.length === 0) {
			rowHeights = Array($chunk.keyidxs.length).fill(20); // same as text-sm
		}
	});

	let columnWidths: Array<number> = [];
	let columnUnit: string = 'px';
	let rowHeights: Array<number> = [];
	let rowUnit: string = 'px';

	let resizeProps = {
		direction: 'x',
		idxBeingResized: -1,
		mouseStart: 0,
		sizeStart: 0,
		offset: 0
	};

	let resizeMethods = {
		mousedown(direction: string, idx: number) {
			return (e: MouseEvent) => {
				// Store the current state in the resize props
				resizeProps.direction = direction;
				resizeProps.idxBeingResized = idx;
				resizeProps.mouseStart = direction === 'x' ? e.clientX : e.clientY;
				resizeProps.sizeStart = direction === 'x' ? columnWidths[idx] : rowHeights[idx];

				// Attach listeners for events
				window.addEventListener('mousemove', resizeMethods.mousemove);
				window.addEventListener('mouseup', resizeMethods.mouseup);
			};
		},

		mousemove(e: MouseEvent) {
			if (resizeProps.idxBeingResized === -1) return;

			// Determine how far the mouse has been moved
			resizeProps.offset =
				(resizeProps.direction === 'x' ? e.clientX : e.clientY) - resizeProps.mouseStart;

			// Update the size
			const newSize = resizeProps.sizeStart + resizeProps.offset;
			if (newSize > 5) {
				if (resizeProps.direction === 'x') columnWidths[resizeProps.idxBeingResized] = newSize;
				else rowHeights[resizeProps.idxBeingResized] = newSize;
			}
		},

		mouseup(e: MouseEvent) {
			window.removeEventListener('mousemove', resizeMethods.mousemove);
			window.removeEventListener('mouseup', resizeMethods.mouseup);
		}
	};

	function onRowClick(e: MouseEvent, keyidx: string) {
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
	}
</script>

<!-- FIXME: Figure out how to do h-full -->
<div class={'rounded-b-md overflow-hidden border-slate-300 ' + classes}>
	<!-- Table -->
	<div
		class={`grid overflow-x-scroll text-sm bg-slate-100 ` + classes}
		style={`grid-template-rows: 1fr ${rowHeights.join(rowUnit + ' ')}${rowUnit}; ` +
			`grid-template-columns: 1fr ${columnWidths.join(columnUnit + ' ')}${columnUnit};`}
	>
		<!-- Header row -->

		<!-- Empty cell for posidx column -->
		<div class="header-cell sticky top-0" />

		{#each $schema.columns as column, col_index}
			<div class="header-cell sticky top-0 flex" style={`grid-column:${col_index + 2} / span 1`}>
				<!-- Column icon and name -->
				<div class="flex items-center gap-1 px-0.5 overflow-hidden">
					<div class="w-5 mr-0.5">
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
					</div>
					<div class="">{column.name}</div>
				</div>

				<!-- Column resizer -->
				<div
					class="absolute flex items-center opacity-0 hover:opacity-100 h-full w-2.5 -right-1.5 cursor-col-resize"
					on:mousedown|preventDefault={resizeMethods.mousedown('x', col_index)}
					on:dblclick|preventDefault={() => {
						columnWidths[col_index] = 100;
					}}
				>
					<div class="flex justify-between h-5 w-full">
						<div class="bg-slate-700 rounded-md" style="width: 3px;" />
						<div class="bg-slate-700 rounded-md" style="width: 3px;" />
					</div>
				</div>
			</div>
		{/each}

		<!-- Data rows -->

		{#each zip($chunk.keyidxs, $chunk.posidxs) as [keyidx, posidx], rowi}
			<!-- First column shows the poxidx (row number) -->
			<div class="header-cell sticky left-0" style={`grid-column: 1 / 2`}>
				<button
					class="w-7 text-center"
					class:text-violet-600={selected.includes(keyidx)}
					class:bg-slate-200={selected.includes(keyidx)}
					on:dblclick={(e) => {
						open_row_modal(posidx);
					}}
					on:click={(e) => onRowClick(e, keyidx)}
				>
					{posidx}
				</button>

				<!-- Row resizer -->
				<div
					class="absolute flex justify-center opacity-0 hover:opacity-100 h-2.5 w-full -bottom-1.5 cursor-row-resize"
					on:mousedown|preventDefault={resizeMethods.mousedown('y', posidx)}
					on:dblclick|preventDefault={() => {
						rowHeights[posidx] = 20;
					}}
				>
					<div class="flex flex-col justify-between w-5 h-full">
						<div class="bg-slate-700 rounded-md" style="height: 3px;" />
						<div class="bg-slate-700 rounded-md" style="height: 3px;" />
					</div>
				</div>
			</div>

			<!-- Data columns -->
			{#each $chunk.columnInfos as col}
				<div class="border-t border-l border-slate-200 hover:opacity-80 bg-white pl-1">
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
		{/each}
	</div>

	<!-- Footer -->
	<div
		class="fixed bottom-0 w-full flex justify-between h-8 z-10 bg-slate-100 px-5 rounded-b-sm border-t border-t-slate-300"
	>
		<div class="px-2 flex space-x-1 items-center">
			{#if selected.length > 0}
				{#if selected.length === 1}
					<Check class="text-violet-600" />
				{:else}
					<CheckAll class="text-violet-600" />
				{/if}
				<div class="text-violet-600 font-mono text-sm ">{selected.length} Selected</div>
			{/if}
		</div>

		<div class="px-2 flex space-x-1 items-center">
			#TODO Wrap: {wrap}
		</div>

		<Pagination bind:page bind:perPage totalItems={$schema.nrows} dropdownPlacement={'top'} />
	</div>
</div>

<style>
	.header-cell {
		@apply border border-slate-300 font-mono text-slate-800;
	}
</style>

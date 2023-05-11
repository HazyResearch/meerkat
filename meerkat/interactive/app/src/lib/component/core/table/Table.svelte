<script lang="ts">
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema, dispatch } from '$lib/utils/api';
	import { DataFrameChunk, type DataFrameRef, type DataFrameSchema } from '$lib/utils/dataframe';
	import { without } from 'underscore';
	import { openModal } from 'svelte-modals';
	import type { Endpoint } from '$lib/utils/types';
	import { zip } from 'underscore';
	import { writable, type Writable } from 'svelte/store';
	import Cell from '$lib/shared/cell/Cell.svelte';
	import { Check, CheckAll, KeyFill } from 'svelte-bootstrap-icons';

	export let df: DataFrameRef;

	export let page: number = 0;
	export let perPage: number = 50;

	export let onEdit: Endpoint;

	export let primarySelectedCell: Array<string> = [];
	export let selectedCells: Array<Array<string>> = [];
	export let selectedCols: Array<string> = [];
	export let selectedRows: Array<string> = [];
	export let onSelectCells: Endpoint;
	export let onSelectCols: Endpoint;
	export let onSelectRows: Endpoint;

	export let classes: string = 'h-fit';
	let wrap: string = 'clip'; // 'wrap' | 'clip' (add 'overflow' later)

	let cutoutWidth: number = 0;
	let cutoutHeight: number = 0;

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
			columnWidths = Array(newSchema.columns.length).fill(100);
		}
	});

	$: fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		formatter: 'tiny'
	}).then((newChunk) => {
		console.log('Fetching chunk');
		chunk.set(newChunk);
		if (rowHeights.length === 0) {
			rowHeights = Array(newChunk.keyidxs.length).fill(22); // same as text-sm + 2
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

	const resizeMethods = {
		mousedown(direction: string, idx: number) {
			return (e: MouseEvent) => {
				// Store the current state in the resize props
				resizeProps.direction = direction;
				resizeProps.idxBeingResized = idx;
				resizeProps.mouseStart = direction === 'x' ? e.x : e.y;
				resizeProps.sizeStart = direction === 'x' ? columnWidths[idx] : rowHeights[idx];

				// Attach listeners for events
				window.addEventListener('mousemove', resizeMethods.mousemove);
				window.addEventListener('mouseup', resizeMethods.mouseup);
			};
		},

		mousemove(e: MouseEvent) {
			if (resizeProps.idxBeingResized === -1) return;

			// Determine how far the mouse has been moved
			resizeProps.offset = (resizeProps.direction === 'x' ? e.x : e.y) - resizeProps.mouseStart;

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

	const selectCellMethods = {
		mousedown(colName: string, keyidx: string) {
			return (e: MouseEvent) => {
				if (e.shiftKey) {
					selectedCells = [];

					// loop through all cells between primarySelectedCell and this cell
					const col1 = $schema.columns.findIndex((c) => c.name === primarySelectedCell[0]);
					const keyidx1 = $chunk.keyidxs.indexOf(primarySelectedCell[1]);

					const col2 = $schema.columns.findIndex((c) => c.name === colName);
					const keyidx2 = $chunk.keyidxs.indexOf(keyidx);

					const [colStart, colEnd] = col1 < col2 ? [col1, col2] : [col2, col1];
					const [keyidxStart, keyidxEnd] =
						keyidx1 < keyidx2 ? [keyidx1, keyidx2] : [keyidx2, keyidx1];
					for (let i = colStart; i <= colEnd; i++) {
						for (let j = keyidxStart; j <= keyidxEnd; j++) {
							selectedCells.push([$schema.columns[i].name, $chunk.keyidxs[j]]);
						}
					}
				} else if (e.metaKey) {
					if (selectedCells.length === 0 && primarySelectedCell.length !== 0) {
						selectedCells = [primarySelectedCell];
					}
					primarySelectedCell = [colName, keyidx];

					const i = selectedCells.findIndex((cell) => cell[0] === colName && cell[1] === keyidx);
					if (i !== -1) selectedCells.splice(i, 1);
					else selectedCells.push([colName, keyidx]);
				} else {
					primarySelectedCell = [colName, keyidx];
					selectedCells = []; // don't add to selectedCells
					selectedCols = [];
					selectedRows = [];
				}
				selectedCells = selectedCells.slice(); // trigger update

				// Attach listeners for events
				window.addEventListener('mousemove', selectCellMethods.mousemove);
				window.addEventListener('mouseup', selectCellMethods.mouseup);
			};
		},

		mousemove(e: MouseEvent) {
			const elements = document.elementsFromPoint(e.x, e.y);
			// loop through elements to find the first cell
			for (let i = 0; i < elements.length; i++) {
				if (elements[i].classList.contains('cell')) {
					const colName = elements[i].getAttribute('colName');
					const keyidx = elements[i].getAttribute('keyidx') || '';

					selectedCells = [];

					// loop through all cells between primarySelectedCell and this cell
					const col1 = $schema.columns.findIndex((c) => c.name === primarySelectedCell[0]);
					const keyidx1 = $chunk.keyidxs.indexOf(primarySelectedCell[1]);

					const col2 = $schema.columns.findIndex((c) => c.name === colName);
					const keyidx2 = $chunk.keyidxs.findIndex((k) => k.toString() === keyidx);

					const [colStart, colEnd] = col1 < col2 ? [col1, col2] : [col2, col1];
					const [keyidxStart, keyidxEnd] =
						keyidx1 < keyidx2 ? [keyidx1, keyidx2] : [keyidx2, keyidx1];
					for (let i = colStart; i <= colEnd; i++) {
						for (let j = keyidxStart; j <= keyidxEnd; j++) {
							selectedCells.push([$schema.columns[i].name, $chunk.keyidxs[j]]);
						}
					}
					break;
				}
			}
		},

		mouseup(e: MouseEvent) {
			const s = selectedCells.length > 0 ? selectedCells : [primarySelectedCell];
			if (onSelectCells && onSelectCells.endpointId) {
				dispatch(onSelectCells.endpointId, { detail: { selected: s } });
			}

			window.removeEventListener('mousemove', selectCellMethods.mousemove);
			window.removeEventListener('mouseup', selectCellMethods.mouseup);
		}
	};

	function onClickCol(e: MouseEvent, colName: string) {
		if (e.shiftKey) {
			if (selectedCols.length === 0) {
				selectedCols.push(colName);
			} else {
				selectedCols = [];
				// loop through all cols between primarySelectedCol and this col
				const col1 = $schema.columns.findIndex((c) => c.name === primarySelectedCell[0]);
				const col2 = $schema.columns.findIndex((c) => c.name === colName);
				const [colStart, colEnd] = col1 < col2 ? [col1, col2] : [col2, col1];
				for (let i = colStart; i <= colEnd; i++) {
					selectedCols.push($schema.columns[i].name);
				}
			}
		} else if (e.metaKey) {
			const i = selectedCols.indexOf(colName);
			if (i !== -1) {
				// remove cells from selectedCells in this col
				selectedCells = selectedCells.filter((cell) => cell[0] !== colName);
				selectedCols.splice(i, 1);
			} else {
				primarySelectedCell = [colName, $chunk.keyidxs[0]];
				selectedCols.push(colName);
			}
		} else {
			primarySelectedCell = [colName, $chunk.keyidxs[0]];
			selectedCells = [];
			selectedCols = [colName];
			selectedRows = [];
		}
		selectedCols = selectedCols.sort(
			(a, b) =>
				$schema.columns.findIndex((c) => c.name === a) -
				$schema.columns.findIndex((c) => c.name === b)
		);

		if (onSelectCols && onSelectCols.endpointId) {
			dispatch(onSelectCols.endpointId, { detail: { selected: selectedCols } });
		}
	}

	function onClickRow(e: MouseEvent, keyidx: string) {
		if (e.shiftKey) {
			if (selectedRows.length === 0) {
				selectedRows.push(keyidx);
			} else {
				selectedRows = [];
				// loop through all rows between primarySelectedCol and this row
				const row1 = $chunk.keyidxs.indexOf(primarySelectedCell[1]);
				const row2 = $chunk.keyidxs.indexOf(keyidx);
				const [rowStart, rowEnd] = row1 < row2 ? [row1, row2] : [row2, row1];
				for (let i = rowStart; i <= rowEnd; i++) {
					selectedRows.push($chunk.keyidxs[i]);
				}
			}
		} else if (e.metaKey) {
			const i = selectedRows.indexOf(keyidx);
			if (i !== -1) {
				// remove cells from selectedCells in this row
				selectedCells = selectedCells.filter((cell) => cell[1] !== keyidx);
				selectedRows.splice(i, 1);
			} else {
				primarySelectedCell = [$schema.columns[0].name, keyidx];
				selectedRows.push(keyidx);
			}
		} else {
			primarySelectedCell = [$schema.columns[0].name, keyidx];
			selectedCells = [];
			selectedCols = [];
			selectedRows = [keyidx];
		}
		selectedRows = selectedRows.sort(
			(a, b) => $chunk.keyidxs.indexOf(a) - $chunk.keyidxs.indexOf(b)
		);

		if (onSelectRows && onSelectRows.endpointId) {
			dispatch(onSelectRows.endpointId, { detail: { selected: selectedCols } });
		}
	}

	function getColumnSelectClasses(
		colName: string,
		primarySelectedCell: Array<string>,
		selectedCells: Array<Array<string>>,
		selectedCols: Array<string>,
		selectedRows: Array<string>
	) {
		if (selectedCols.includes(colName))
			return 'bg-violet-700 text-white font-bold '
		if (
			primarySelectedCell[0] === colName ||
			selectedCells.some((c) => c[0] === colName) ||
			selectedRows.length > 0
		)
			return 'bg-violet-100 ';
		return '';
	}

	function getRowSelectClasses(
		keyidx: string,
		primarySelectedCell: Array<string>,
		selectedCells: Array<Array<string>>,
		selectedCols: Array<string>,
		selectedRows: Array<string>
	) {
		if (selectedRows.includes(keyidx))
			return 'bg-violet-700 text-white font-bold '
		if (
			primarySelectedCell[1] === keyidx ||
			selectedCells.some((c) => c[1] === keyidx) ||
			selectedCols.length > 0
		)
			return 'bg-violet-100 ';
		return '';
	}

	function getCellSelectClasses(
		colName: string,
		keyidx: string,
		primarySelectedCell: Array<string>,
		selectedCells: Array<Array<string>>,
		selectedCols: Array<string>,
		selectedRows: Array<string>
	) {
		let classes = '';
		if (
			selectedCells.some((c) => c[0] === colName && c[1] === keyidx) ||
			selectedCols.includes(colName) ||
			selectedRows.includes(keyidx)
		)
			classes += 'bg-violet-100 ';
		if (primarySelectedCell[0] === colName && primarySelectedCell[1] === keyidx)
			classes += 'border-2 border-violet-600 ';
		else classes += 'border-t border-l border-slate-300 ';
		return classes;
	}
</script>

<!-- Subtract 15px for scrollbar -->
<div
	class={'rounded-b-md border-slate-300 w-fit ' + classes}
	style="max-width:calc(100vw - 15px); max-height:calc(100vh - 15px)"
>
	<!-- Table (max-height subtracts 32px for height of footer)-->
	<div
		class={`grid overflow-x-scroll overflow-y-scroll text-sm border border-slate-300`}
		style={`grid-template-rows: 1fr ${rowHeights.join(rowUnit + ' ')}${rowUnit}; ` +
			`grid-template-columns: min-content ${columnWidths.join(columnUnit + ' ')}${columnUnit};` +
			'max-height:calc(100vh - 32px)'}
	>
		<!-- Header row -->

		<!-- Placeholder for cutout in top left corner to hide scrolling headers -->
		<div
			class="header-cell border border-slate-300 font-mono bg-slate-100 text-slate-800"
			style={`grid-column:1 / 2; grid-row:1 / 2;`}
			bind:clientWidth={cutoutWidth}
			bind:clientHeight={cutoutHeight}
		/>

		{#each $schema.columns as column, col_index}
			<div
				class={'header-cell border border-slate-300 font-mono bg-slate-100 text-slate-800 sticky top-0 z-10 flex ' +
					getColumnSelectClasses(
						column.name,
						primarySelectedCell,
						selectedCells,
						selectedCols,
						selectedRows
					)}
				style={`grid-column:${col_index + 2} / span 1`}
			>
				<!-- Column icon and name -->
				<button
					class="flex items-center gap-1 px-0.5 overflow-hidden"
					on:click={(e) => onClickCol(e, column.name)}
				>
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
				</button>

				<!-- Column resizer -->
				<div
					class="absolute flex items-center opacity-0 hover:opacity-100 h-full w-1 right-0 cursor-col-resize"
					on:mousedown|preventDefault={resizeMethods.mousedown('x', col_index)}
					on:dblclick|preventDefault={() => {
						columnWidths[col_index] = 100;
					}}
				>
					<div class="flex justify-between h-5 w-full">
						<div class="bg-slate-700 rounded-md" style="width: 3px;" />
					</div>
				</div>
			</div>
		{/each}

		<!-- Data rows -->

		{#each zip($chunk.keyidxs, $chunk.posidxs) as [keyidx, posidx], rowi}
			<!-- First column shows the posidx (row number) -->
			<div
				class="header-cell border border-slate-300 font-mono bg-slate-100 text-slate-800 sticky left-0 z-10"
				style={`grid-column: 1 / 2`}
			>
				<button
					class={'w-7 text-center ' +
						getRowSelectClasses(
							keyidx,
							primarySelectedCell,
							selectedCells,
							selectedCols,
							selectedRows
						)}
					on:dblclick={(e) => open_row_modal(posidx)}
					on:click|preventDefault={(e) => onClickRow(e, keyidx)}
				>
					{posidx}
				</button>

				<!-- Row resizer -->
				<div
					class="absolute flex justify-center opacity-0 hover:opacity-100 h-1 w-full bottom-0 cursor-row-resize"
					on:mousedown|preventDefault={resizeMethods.mousedown('y', posidx)}
					on:dblclick|preventDefault={() => {
						rowHeights[posidx] = 20;
					}}
				>
					<div class="flex flex-col justify-between w-5 h-full">
						<div class="bg-slate-700 rounded-md" style="height: 3px;" />
					</div>
				</div>
			</div>

			<!-- Data columns -->
			{#each $chunk.columnInfos as col}
				<!-- on:click={(e) => {
					onClickCell(e, col.name, keyidx);
					document.getSelection().removeAllRanges();
				}} -->
				<div
					class={'cell bg-white pl-1 ' +
						getCellSelectClasses(
							col.name,
							keyidx,
							primarySelectedCell,
							selectedCells,
							selectedCols,
							selectedRows
						)}
					on:mousedown|preventDefault={selectCellMethods.mousedown(col.name, keyidx)}
					colName={col.name}
					{keyidx}
				>
					<Cell
						{...$chunk.getCell(rowi, col.name)}
						editable={true && false}
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

	<!-- Cutout in top left corner to hide scrolling headers -->
	<div
		class="header-cell border border-slate-300 font-mono bg-slate-100 text-slate-800 absolute top-px left-px z-20"
		style={`width:${cutoutWidth + 2}px; height:${cutoutHeight + 2}px`}
	/>

	<!-- Footer -->
	<div
		class="flex justify-between h-8 z-10 bg-slate-100 px-5 rounded-b-sm border-t border-t-slate-300"
	>
		<div class="px-2 flex space-x-1 items-center">
			{#if selectedRows.length > 0}
				{#if selectedRows.length === 1}
					<Check class="text-violet-600" />
				{:else}
					<CheckAll class="text-violet-600" />
				{/if}
				<div class="text-violet-600 font-mono text-sm ">{selectedRows.length} rows selected</div>
			{/if}
		</div>

		<!-- <div class="px-2 flex space-x-1 items-center">
			#TODO Wrap: {wrap}
		</div> -->

		<Pagination bind:page bind:perPage totalItems={$schema.nrows} dropdownPlacement={'top'} />
	</div>
</div>

<style>
	/* .header-cell {
		@apply border border-slate-300 font-mono bg-slate-100 text-slate-800;
	} */
</style>

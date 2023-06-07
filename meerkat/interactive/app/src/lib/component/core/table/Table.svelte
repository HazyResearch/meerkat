<script lang="ts">
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema, dispatch } from '$lib/utils/api';
	import { DataFrameChunk, type DataFrameRef, type DataFrameSchema } from '$lib/utils/dataframe';
	import { openModal } from 'svelte-modals';
	import type { Endpoint } from '$lib/utils/types';
	import { zip } from 'underscore';
	import { writable, type Writable } from 'svelte/store';
	import Cell from '$lib/shared/cell/Cell.svelte';
	import { Check, CheckAll, KeyFill } from 'svelte-bootstrap-icons';

	type Cell = {
		column: string;
		keyidx: number; // TODO: decide if this should be number or string
		posidx: number;
		value: any;
	};

	export let df: DataFrameRef;

	export let page: number = 0;
	export let perPage: number = 50;

	let editMode: boolean = false;
	let editValue: string = '';
	export let onEdit: Endpoint;

	let primarySelectedCell: Cell = { column: '', keyidx: -1, posidx: -1, value: '' };
	let secondarySelectedCell: Cell = { column: '', keyidx: -1, posidx: -1, value: '' };
	let activeCells: Array<Cell> = []; // the cells currently being interacted with
	let selectedCells: Array<Cell> = [];
	let selectedCols: Array<string> = [];
	let selectedRows: Array<number> = [];
	export let onSelectCells: Endpoint;
	export let onSelectCols: Endpoint;
	export let onSelectRows: Endpoint;

	export let classes: string = 'h-fit';
	let wrapping: string = 'clip';

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

	let columnWidths: Array<number> = [];
	let columnUnit: string = 'px';
	let rowHeights: Array<number> = [];
	let rowUnit: string = 'px';

	function setColumnWidths(clamp: number | null = null) {
		const headerCells = document.getElementsByClassName('header-cell');
		// Use i + 1 because the first header-cell is the top-left cutout
		columnWidths = columnWidths.map((_, i) =>
			clamp
				? Math.min((headerCells[i + 1] as HTMLElement).offsetWidth, clamp)
				: (headerCells[i + 1] as HTMLElement).offsetWidth
		);
	}

	function setRowHeights() {
		const headerCells = document.getElementsByClassName('header-cell');
		// Use i + 1 because the first header-cell is the top-left cutout
		rowHeights = rowHeights.map(
			(_, i) => (headerCells[i + 1 + columnWidths.length] as HTMLElement).offsetHeight
		);
	}

	// Create placeholder variables for table data.
	let schema: Writable<DataFrameSchema> = writable({
		columns: [],
		primaryKey: 'pkey',
		nrows: 0,
		id: ''
	});
	let chunk: Writable<DataFrameChunk> = writable(new DataFrameChunk([], [], [], 0, 'pkey'));

	// Run this once to set columnWidths
	fetchSchema({
		df: df,
		formatter: 'icon'
	}).then((newSchema) => {
		columnWidths = Array(newSchema.columns.length).fill(-1);
		// After giving 2 seconds for the table to load, clamp the column widths
		setTimeout(() => setColumnWidths(300), 2000);
	});

	$: fetchSchema({
		df: df,
		formatter: 'icon'
	}).then((newSchema) => {
		schema.set(newSchema);
		// After giving 2 seconds for the table to load, set the row heights
		setTimeout(setRowHeights, 2000);
	});

	// Run this once to set rowHeights (24 is same as text-sm + 4)
	fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		formatter: 'tiny'
	}).then((newChunk) => (rowHeights = Array(newChunk.keyidxs.length).fill(-1)));

	$: fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		formatter: 'tiny'
	}).then((newChunk) => chunk.set(newChunk));

	let resizeProps = {
		direction: 'x',
		idxBeingResized: -1,
		mouseStart: 0,
		sizeStart: 0,
		offset: 0
	};

	/**
	 * Helper function to find the first ancestor of an element with a given
	 * class.
	 *
	 * @param et - The event target
	 * @param cls - The class to search for
	 */
	const findAncestor = (et: EventTarget | null, cls: string) => {
		let el: HTMLElement | null = et as HTMLElement;
		while (el && (el = el.parentElement) && !el.classList.contains(cls));
		return el;
	};

	const resizeMethods = {
		mousedown(direction: string, idx: number) {
			return (e: MouseEvent) => {
				// Store the current state in the resize props
				resizeProps.direction = direction;
				resizeProps.idxBeingResized = idx;
				resizeProps.mouseStart = direction === 'x' ? e.x : e.y;

				const el = findAncestor(e.target, 'header-cell');
				if (el) {
					if (direction === 'x') columnWidths[idx] = el.offsetWidth;
					else rowHeights[idx] = el.offsetHeight;
				}
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

	/**
	 * Helper function to convert a column name to the index of that column in
	 * the schema.
	 * @param col - The name of the column
	 */
	function col2idx(col: string) {
		return $schema.columns.findIndex((c) => c.name === col);
	}

	/**
	 * Helper function to convert a keyidx to the index of that keyidx in the
	 * chunk. This also corresponds to the posidx.
	 * @param keyidx - The keyidx of the row
	 */
	function keyidx2idx(keyidx: number) {
		// TODO: figure out why we need to do k.toString(). It is a number when it should be a string
		return $chunk.keyidxs.findIndex((k) => k.toString() === keyidx.toString());
	}

	/**
	 * Helper function to get the cell at a given column and posidx.
	 * @param col
	 * @param posidx
	 */
	function getCell(column: string, posidx: number) {
		return {
			column,
			keyidx: parseInt($chunk.keyidxs[posidx]),
			posidx,
			value: $chunk.getCell(posidx, column).data
		};
	}

	/**
	 * Helper function to determine if two cells are equal.
	 * @param cell1
	 * @param cell2
	 */
	function areEqual(cell1: Cell, cell2: Cell) {
		return cell1.column === cell2.column && cell1.posidx === cell2.posidx;
	}

	/**
	 * Selects all cells between cell1 and cell2, inclusive.
	 * @param cell1
	 * @param cell2
	 * @param select - If true, will select the cells. If false, will mark as active
	 */
	function selectRange(cell1: Cell, cell2: Cell, select = true) {
		if (select) selectedCells = [];
		else activeCells = [];

		const [col1, row1] = [col2idx(cell1.column), cell1.posidx];
		const [col2, row2] = [col2idx(cell2.column), cell2.posidx];

		// If the user clicks on the same cell, none should be selected
		if (col1 !== -1 && col1 === col2 && row1 !== -1 && row1 === row2) return;

		const [colStart, colEnd] = col1 < col2 ? [col1, col2] : [col2, col1];
		const [keyidxStart, keyidxEnd] = row1 < row2 ? [row1, row2] : [row2, row1];

		for (let i = colStart; i <= colEnd; i++) {
			for (let j = keyidxStart; j <= keyidxEnd; j++) {
				const column = $schema.columns[i].name;
				const keyidx = parseInt($chunk.keyidxs[j]);
				const posidx = j;
				const value = $chunk.getCell(posidx, column).data;
				if (select) selectedCells.push({ column, keyidx, posidx, value });
				else activeCells.push({ column, keyidx, posidx, value });
			}
		}

		// trigger update
		if (select) selectedCells = selectedCells.slice();
		else activeCells = activeCells.slice();
	}

	const selectCellMethods = {
		mousedown(cell: Cell) {
			if (editMode) {
				if (onEdit && onEdit.endpointId) {
					const { column, keyidx, posidx } = primarySelectedCell;
					primarySelectedCell.value = editValue;
					dispatch(onEdit.endpointId, {
						detail: {
							column,
							keyidx,
							posidx,
							value: editValue
						}
					});
				}
				editMode = false;
			}
			return (e: MouseEvent) => {
				if (e.shiftKey) {
					secondarySelectedCell = cell;
					selectRange(primarySelectedCell, secondarySelectedCell);
				} else if (e.metaKey) {
					if (getSelectedBitmap(primarySelectedCell.column, primarySelectedCell.keyidx) === 0)
						selectedCells.push(primarySelectedCell);
					primarySelectedCell = secondarySelectedCell = cell;
					activeCells.push(cell);
				} else {
					primarySelectedCell = secondarySelectedCell = cell;
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
			// Loop through elements beneath the mouse x and y to find the first
			// element with class 'cell'
			for (const element of document.elementsFromPoint(e.x, e.y)) {
				if (!element.classList.contains('cell')) {
					continue;
				}
				const column = element.getAttribute('column') || '';
				const keyidx = parseInt(element.getAttribute('keyidx') || '-1');

				if (
					selectedCells.length > 0 &&
					column === primarySelectedCell.column &&
					keyidx === primarySelectedCell.keyidx
				) {
					return;
				}

				const posidx = keyidx2idx(keyidx);
				secondarySelectedCell = getCell(column, posidx);
				selectRange(primarySelectedCell, secondarySelectedCell, false);

				break;
			}
		},

		mouseup(e: MouseEvent) {
			// select all activeCells if at least one of them is not already selected
			let foundUnselected = false;
			for (const cell of activeCells) {
				if (getSelectedBitmap(cell.column, cell.keyidx, false) === 0) {
					foundUnselected = true;
					break;
				}
			}

			if (foundUnselected) {
				// select all cells
				selectedCells = selectedCells.concat(activeCells);
			} else {
				// unselect all cells
				for (const cell of activeCells) {
					const i = selectedCells.findIndex((c) => areEqual(c, cell));
					if (i !== -1) {
						if (selectedCells.length === 2) {
							selectedCells.splice(i, 1);
							primarySelectedCell = secondarySelectedCell = selectedCells[0];
							selectedCells = [];
						} else if (
							selectedCells.length > 1 &&
							(areEqual(selectedCells[i], primarySelectedCell) ||
								areEqual(selectedCells[i], secondarySelectedCell))
						) {
							selectedCells.splice(i, 1);
							primarySelectedCell = secondarySelectedCell = selectedCells[0];
						} else {
							selectedCells.splice(i, 1);
						}
					} else if (selectedCols.includes(cell.column)) {
						selectedCols.splice(selectedCols.indexOf(cell.column), 1);
						// add all the cells in the column to selectedCells except the one we clicked on
						for (let i = 0; i < $chunk.keyidxs.length; i++) {
							if (i !== cell.posidx) {
								selectedCells.push(getCell(cell.column, i));
							}
						}
						if (selectedCells.length > 0)
							primarySelectedCell = secondarySelectedCell = selectedCells[0];
						else primarySelectedCell = secondarySelectedCell = getCell(cell.column, cell.posidx);
					} else if (selectedRows.includes(cell.keyidx)) {
						selectedRows.splice(selectedRows.indexOf(cell.keyidx), 1);
						// add all the cells in the row to selectedCells except the one we clicked on
						for (let i = 0; i < $schema.columns.length; i++) {
							if ($schema.columns[i].name !== cell.column) {
								selectedCells.push(getCell($schema.columns[i].name, cell.posidx));
							}
						}
						if (selectedCells.length > 0)
							primarySelectedCell = secondarySelectedCell = selectedCells[0];
						else primarySelectedCell = secondarySelectedCell = getCell(cell.column, cell.posidx);
					}
				}
			}

			activeCells = [];

			let s = [primarySelectedCell];
			if (selectedCells.length > 0) {
				// filter out duplicate cells
				s = selectedCells.filter((cell, i, arr) => arr.findIndex((c) => areEqual(c, cell)) === i);
			}
			if (onSelectCells && onSelectCells.endpointId) {
				dispatch(onSelectCells.endpointId, { detail: { selected: s } });
			}

			window.removeEventListener('mousemove', selectCellMethods.mousemove);
			window.removeEventListener('mouseup', selectCellMethods.mouseup);
		}
	};

	function onClickCol(e: MouseEvent, column: string) {
		if (e.shiftKey) {
			selectedCols = [];
			// loop through all cols between primarySelectedCol and this col
			const col1 = col2idx(primarySelectedCell.column);
			const col2 = col2idx(column);
			const [colStart, colEnd] = col1 < col2 ? [col1, col2] : [col2, col1];
			for (let i = colStart; i <= colEnd; i++) {
				selectedCols.push($schema.columns[i].name);
			}
		} else if (e.metaKey) {
			const i = selectedCols.indexOf(column);
			if (i !== -1) {
				// remove cells from selectedCells in this col
				selectedCells = selectedCells.filter((c) => c.column !== column);
				selectedCols.splice(i, 1);

				// set new primarySelectedCell
				if (selectedCells.length > 0)
					primarySelectedCell = secondarySelectedCell = selectedCells[0];
				else if (selectedCols.length > 0)
					primarySelectedCell = secondarySelectedCell = getCell(selectedCols[0], 0);
				else if (selectedRows.length > 0)
					primarySelectedCell = secondarySelectedCell = getCell(
						$schema.columns[0].name,
						selectedRows[0]
					);
				else primarySelectedCell = secondarySelectedCell = getCell($schema.columns[0].name, 0);
			} else {
				primarySelectedCell = secondarySelectedCell = getCell(column, 0);
				selectedCols.push(column);
			}
		} else {
			primarySelectedCell = secondarySelectedCell = getCell(column, 0);
			selectedCells = [];
			selectedCols = [column];
			selectedRows = [];
		}
		selectedCols = selectedCols.sort((a, b) => col2idx(a) - col2idx(b));

		if (onSelectCols && onSelectCols.endpointId) {
			dispatch(onSelectCols.endpointId, { detail: { selected: selectedCols } });
		}
	}

	function onClickRow(e: MouseEvent, keyidx: number) {
		const posidx = keyidx2idx(keyidx);
		if (e.shiftKey) {
			selectedRows = [];
			// loop through all rows between primarySelectedCol and this row
			const row1 = primarySelectedCell.posidx;
			const row2 = posidx;
			const [rowStart, rowEnd] = row1 < row2 ? [row1, row2] : [row2, row1];
			for (let i = rowStart; i <= rowEnd; i++) {
				selectedRows.push(parseInt($chunk.keyidxs[i]));
			}
		} else if (e.metaKey) {
			const i = selectedRows.indexOf(keyidx);
			if (i !== -1) {
				// remove cells from selectedCells in this row
				selectedCells = selectedCells.filter((c) => c.keyidx !== keyidx);
				selectedRows.splice(i, 1);

				// set new primarySelectedCell
				if (selectedCells.length > 0)
					primarySelectedCell = secondarySelectedCell = selectedCells[0];
				else if (selectedCols.length > 0)
					primarySelectedCell = secondarySelectedCell = getCell(selectedCols[0], 0);
				else if (selectedRows.length > 0)
					primarySelectedCell = secondarySelectedCell = getCell(
						$schema.columns[0].name,
						selectedRows[0]
					);
				else primarySelectedCell = secondarySelectedCell = getCell($schema.columns[0].name, 0);
			} else {
				const column = $schema.columns[0].name;
				primarySelectedCell = secondarySelectedCell = getCell(column, posidx);
				selectedRows.push(keyidx);
			}
		} else {
			const column = $schema.columns[0].name;
			primarySelectedCell = secondarySelectedCell = getCell(column, posidx);
			selectedCells = [];
			selectedCols = [];
			selectedRows = [keyidx];
		}
		selectedRows = selectedRows.sort((a, b) => keyidx2idx(a) - keyidx2idx(b));

		if (onSelectRows && onSelectRows.endpointId) {
			dispatch(onSelectRows.endpointId, { detail: { selected: selectedCols } });
		}
	}

	function getColumnSelectClasses(
		column: string,
		primarySelectedCell: Cell,
		activeCells: Array<Cell>,
		selectedCells: Array<Cell>,
		selectedCols: Array<string>,
		selectedRows: Array<number>
	) {
		if (selectedCols.includes(column)) return 'bg-violet-700 text-white font-bold ';
		if (
			primarySelectedCell.column === column ||
			activeCells.some((c) => c.column === column) ||
			selectedCells.some((c) => c.column === column) ||
			selectedRows.length > 0
		)
			return 'bg-violet-200 font-bold ';
		return '';
	}

	function getRowSelectClasses(
		keyidx: number,
		primarySelectedCell: Cell,
		activeCells: Array<Cell>,
		selectedCells: Array<Cell>,
		selectedCols: Array<string>,
		selectedRows: Array<number>
	) {
		if (selectedRows.includes(keyidx)) return 'bg-violet-700 text-white font-bold ';
		if (
			primarySelectedCell.keyidx === keyidx ||
			activeCells.some((c) => c.keyidx === keyidx) ||
			selectedCells.some((c) => c.keyidx === keyidx) ||
			selectedCols.length > 0
		)
			return 'bg-violet-200 font-bold ';
		return '';
	}

	/**
	 * Helper function that returns a number representing the ways a cell has
	 * been selected or not.
	 *
	 * - ones's place: selectedCols (0 or 1)
	 * - ten's place: selectedRows (0 or 10)
	 * - hundred's place: activeCells (0 or 100)
	 * - thousand's place: number of occurences in selectedCells
	 *
	 * @param column
	 * @param keyidx
	 * @param countActive
	 */
	function getSelectedBitmap(column: string, keyidx: number, countActive = true) {
		return (
			(selectedCols.includes(column) ? 1 : 0) +
			(selectedRows.includes(keyidx) ? 10 : 0) +
			(countActive && activeCells.some((c) => c.column === column && c.keyidx === keyidx)
				? 100
				: 0) +
			selectedCells.filter((c) => c.column === column && c.keyidx === keyidx).length * 1000
		);
	}

	/**
	 * Helper function that returns the number of times a cell is selected.
	 * @param column
	 * @param keyidx
	 * @param posidx
	 * @param primarySelectedCell
	 * @param activeCells
	 * @param selectedCells
	 * @param selectedCols
	 * @param selectedRows
	 */
	function getSelectedCount(column: string, keyidx: number) {
		return (
			(selectedCols.includes(column) ? 1 : 0) +
			(selectedRows.includes(keyidx) ? 1 : 0) +
			(activeCells.some((c) => c.column === column && c.keyidx === keyidx) ? 1 : 0) +
			selectedCells.filter((c) => c.column === column && c.keyidx === keyidx).length
		);
	}

	/**
	 * Helper function that returns a string of classes for a cell based on
	 * how it has been selected.
	 *
	 * NOTE: The params activeCells, selectedCells, selectedCols, and
	 * selectedRows are included so that this funciton is called reactively
	 * whenever any of those (or primarySelectedCell) changes.
	 *
	 * @param column
	 * @param keyidx
	 * @param posidx
	 * @param primarySelectedCell
	 * @param activeCells
	 * @param selectedCells
	 * @param selectedCols
	 * @param selectedRows
	 */
	function getCellSelectClasses(
		column: string,
		keyidx: number,
		posidx: number,
		primarySelectedCell: Cell,
		activeCells: Array<Cell>,
		selectedCells: Array<Cell>,
		selectedCols: Array<string>,
		selectedRows: Array<number>,
		editMode: boolean
	) {
		let classes = '';
		const bitmap = getSelectedBitmap(column, keyidx);

		// TODO: Determine overflow
		// if (wrapping !== 'overflow') classes += 'overflow-hidden ';

		// Determine background color
		if (bitmap > 0) {
			// the max tailwind color is 900, so we cap the count at 9
			classes += `bg-violet-${Math.min(getSelectedCount(column, keyidx), 9)}00 `;
		}

		// Determine borders
		if (primarySelectedCell.column === column && primarySelectedCell.keyidx === keyidx) {
			if (editMode) classes += 'overflow-visible -ml-px -mt-px ';
			else classes += 'border-t-2 border-l-2 border-violet-600 ';
		} else {
			// border width of 1px, default color slate
			classes += 'border-t border-l border-slate-300 ';

			if (posidx > 0) {
				if (areEqual(getCell(column, posidx - 1), primarySelectedCell)) {
					if (!editMode) classes += 'border-t-2 border-t-violet-600 -mt-px ';
				} else {
					const bitmapAbove = getSelectedBitmap(column, parseInt($chunk.keyidxs[posidx - 1]));
					if (bitmap !== bitmapAbove) classes += 'border-t-violet-600 ';
				}
			} else if (bitmap > 0 && posidx === 0) {
				classes += 'border-t-violet-600 ';
			}

			if (bitmap > 0 && posidx === $chunk.keyidxs.length - 1)
				classes += 'border-b border-b-violet-600 ';

			const colidx = col2idx(column);
			if (colidx > 0) {
				if (areEqual(getCell($chunk.columns[colidx - 1], posidx), primarySelectedCell)) {
					if (!editMode) classes += 'border-l-2 border-l-violet-600 -ml-px ';
				} else {
					const bitmapLeft = getSelectedBitmap($chunk.columns[colidx - 1], keyidx);
					if (bitmap !== bitmapLeft) classes += 'border-l-violet-600 ';
				}
			} else if (bitmap > 0 && colidx === 0) {
				classes += 'border-l-violet-600 ';
			}

			if (bitmap > 0 && colidx === $schema.columns.length - 1)
				classes += 'border-r border-r-violet-600 ';
		}

		return classes;
	}

	/**
	 * Start editing the current cell.
	 */
	function startEdit() {
		editMode = true;
		editValue = primarySelectedCell.value;
	}

	/**
	 * Finish editing. If callOnEdit is true, save the edit by calling the
	 * onEdit endpoint.
	 * @param callOnEdit Whether or not to call the onEdit endpoint.
	 */
	function endEdit(callOnEdit: boolean = true) {
		if (callOnEdit && onEdit && onEdit.endpointId) {
			const { column, keyidx, posidx } = primarySelectedCell;
			primarySelectedCell.value = editValue;
			dispatch(onEdit.endpointId, {
				detail: {
					column,
					keyidx,
					posidx,
					value: editValue
				}
			});
		}
		editMode = false;
	}

	// Define keyboard shortcuts
	window.addEventListener('keydown', (e) => {
		if (primarySelectedCell.column === '') {
			console.log('setting to first cell', $schema.columns[0].name);
			primarySelectedCell = secondarySelectedCell = getCell($schema.columns[0].name, 0);
		}

		const colidx = col2idx(primarySelectedCell.column);
		const posidx = primarySelectedCell.posidx;

		if (e.key === 'a' && e.metaKey) {
			e.preventDefault();
			selectedCells = [];
			selectedCols = $schema.columns.map((c) => c.name);
			selectedRows = $chunk.keyidxs.map((k) => parseInt(k));
		} else if (e.key === 'ArrowDown') {
			if (editMode) return;
			e.preventDefault();
			if (e.metaKey) {
				if (e.shiftKey) {
					secondarySelectedCell = getCell(secondarySelectedCell.column, $chunk.keyidxs.length - 1);
					selectRange(primarySelectedCell, secondarySelectedCell);
				} else {
					primarySelectedCell = secondarySelectedCell = getCell(
						primarySelectedCell.column,
						$chunk.keyidxs.length - 1
					);
				}
			} else if (e.shiftKey) {
				const posidx2 = secondarySelectedCell.posidx;
				if (posidx2 + 1 < $chunk.keyidxs.length) {
					secondarySelectedCell = getCell(secondarySelectedCell.column, posidx2 + 1);
					selectRange(primarySelectedCell, secondarySelectedCell);
				} else if (page * perPage + posidx2 < $schema.nrows - 1) {
					// TODO: flesh out this case
					page++;
					secondarySelectedCell = getCell(secondarySelectedCell.column, 0);
				}
			} else {
				if (posidx < $chunk.keyidxs.length - 1) {
					primarySelectedCell = secondarySelectedCell = getCell(
						primarySelectedCell.column,
						posidx + 1
					);
				} else if (page * perPage + posidx < $schema.nrows - 1) {
					page++;
					primarySelectedCell = secondarySelectedCell = getCell(primarySelectedCell.column, 0);
				}
				selectedCells = [];
			}
			selectedCols = [];
			selectedRows = [];
		} else if (e.key === 'ArrowUp') {
			if (editMode) return;
			e.preventDefault();
			if (e.metaKey) {
				if (e.shiftKey) {
					secondarySelectedCell = getCell(secondarySelectedCell.column, 0);
					selectRange(primarySelectedCell, secondarySelectedCell);
				} else {
					primarySelectedCell = secondarySelectedCell = getCell(primarySelectedCell.column, 0);
				}
			} else if (e.shiftKey) {
				const posidx2 = secondarySelectedCell.posidx;
				if (posidx2 > 0) {
					secondarySelectedCell = getCell(secondarySelectedCell.column, posidx2 - 1);
					selectRange(primarySelectedCell, secondarySelectedCell);
				} else if (page > 0) {
					// TODO: flesh out this case
					page--;
					secondarySelectedCell = getCell(secondarySelectedCell.column, $chunk.keyidxs.length - 1);
				}
			} else {
				if (posidx > 0) {
					primarySelectedCell = secondarySelectedCell = getCell(
						primarySelectedCell.column,
						posidx - 1
					);
				} else if (page > 0) {
					page--;
					primarySelectedCell = secondarySelectedCell = getCell(
						primarySelectedCell.column,
						$chunk.keyidxs.length - 1
					);
				}
				selectedCells = [];
			}
			selectedCols = [];
			selectedRows = [];
		} else if (e.key === 'ArrowLeft') {
			if (editMode) return;
			e.preventDefault();
			if (e.metaKey) {
				if (e.shiftKey) {
					secondarySelectedCell = getCell($schema.columns[0].name, secondarySelectedCell.posidx);
					selectRange(primarySelectedCell, secondarySelectedCell);
				} else {
					primarySelectedCell = secondarySelectedCell = getCell($schema.columns[0].name, posidx);
				}
			} else if (e.shiftKey) {
				const colidx2 = col2idx(secondarySelectedCell.column);
				const posidx2 = secondarySelectedCell.posidx;
				if (colidx2 > 0) {
					secondarySelectedCell = getCell($schema.columns[colidx2 - 1].name, posidx2);
					selectRange(primarySelectedCell, secondarySelectedCell);
				}
			} else {
				if (colidx > 0) {
					primarySelectedCell = secondarySelectedCell = getCell(
						$schema.columns[colidx - 1].name,
						posidx
					);
				}
				selectedCells = [];
			}
			selectedCols = [];
			selectedRows = [];
		} else if (e.key === 'ArrowRight') {
			if (editMode) return;
			e.preventDefault();
			if (e.metaKey) {
				if (e.shiftKey) {
					secondarySelectedCell = getCell(
						$schema.columns[$schema.columns.length - 1].name,
						secondarySelectedCell.posidx
					);
					selectRange(primarySelectedCell, secondarySelectedCell);
				} else {
					primarySelectedCell = secondarySelectedCell = getCell(
						$schema.columns[$schema.columns.length - 1].name,
						posidx
					);
				}
			} else if (e.shiftKey) {
				const colidx2 = col2idx(secondarySelectedCell.column);
				const posidx2 = secondarySelectedCell.posidx;
				if (colidx2 < $schema.columns.length - 1) {
					secondarySelectedCell = getCell($schema.columns[colidx2 + 1].name, posidx2);
					selectRange(primarySelectedCell, secondarySelectedCell);
				}
			} else {
				if (colidx < $schema.columns.length - 1) {
					primarySelectedCell = secondarySelectedCell = getCell(
						$schema.columns[colidx + 1].name,
						posidx
					);
				}
				selectedCells = [];
			}
			selectedCols = [];
			selectedRows = [];
		} else if (e.key === 'Tab') {
			e.preventDefault();
			if (editMode) endEdit();
			// TODO: if there are selected cells, loop through them
			if (e.shiftKey) {
				if (colidx > 0) {
					primarySelectedCell = secondarySelectedCell = getCell(
						$schema.columns[colidx - 1].name,
						posidx
					);
				}
			} else {
				if (colidx < $schema.columns.length - 1) {
					primarySelectedCell = secondarySelectedCell = getCell(
						$schema.columns[colidx + 1].name,
						posidx
					);
				}
			}
			selectedCells = [];
			selectedCols = [];
			selectedRows = [];
		} else if (e.key === 'Enter') {
			if (editMode) {
				if (e.ctrlKey || e.altKey || e.metaKey) {
					return;
				} else {
					e.preventDefault();
					endEdit();
				}
			} else {
				e.preventDefault();
				startEdit();
			}
		} else if (e.key === 'Escape') {
			// TODO: make the cell revert immediately instead of on refresh
			e.preventDefault();
			if (editMode) endEdit(false);
		}
	});
</script>

<!-- Subtract 15px for scrollbar -->
<div
	class={'rounded-b-md border-slate-300 w-fit bg-white ' + classes}
	style="max-width:calc(100vw - 15px); max-height:calc(100vh - 15px)"
>
	<!-- Table (max-height subtracts 32px for height of footer)-->
	<div
		class={`grid overflow-x-scroll overflow-y-scroll text-sm border border-slate-300`}
		style={`grid-template-rows: min-content ${rowHeights
			.map((h) => (h === -1 ? 'min-content' : `${h}${rowUnit}`))
			.join(' ')};` +
			`grid-template-columns: min-content ${columnWidths
				.map((w) => (w === -1 ? 'min-content' : `${w}${columnUnit}`))
				.join(' ')};` +
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
				class={'header-cell border border-slate-300 font-mono bg-slate-100 text-slate-800 sticky top-0 z-20 flex ' +
					getColumnSelectClasses(
						column.name,
						primarySelectedCell,
						activeCells,
						selectedCells,
						selectedCols,
						selectedRows
					)}
				style={`grid-column:${col_index + 2} / span 1`}
			>
				<!-- Column icon and name -->
				<button
					class="flex w-full justify-center items-center gap-1 px-0.5 overflow-hidden"
					on:click={(e) => onClickCol(e, column.name)}
				>
					<div class="w-5 mr-0.5">
						{#if column.name === $schema.primaryKey}
							<!-- Show a key icon for the primary key-->
							<KeyFill
								class={selectedCols.includes(column.name) ? 'text-violet-300' : 'text-violet-600'}
							/>
						{:else}
							<!-- TODO: make white when selected -->
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
					on:dblclick|preventDefault={(e) => {
						columnWidths[col_index] = -1; // let it auto-size to min-content
						setTimeout(() => {
							const el = findAncestor(e.target, 'header-cell');
							if (el) columnWidths[col_index] = el.offsetWidth;
						});
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
				class="header-cell border border-slate-300 font-mono bg-slate-100 text-slate-800 sticky left-0 z-20"
				style={`grid-column: 1 / 2`}
			>
				<button
					class={'w-7 h-full text-center ' +
						getRowSelectClasses(
							keyidx,
							primarySelectedCell,
							activeCells,
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
					on:dblclick|preventDefault={(e) => {
						rowHeights[posidx] = -1; // let it auto-size to min-content
						setTimeout(() => {
							const el = findAncestor(e.target, 'header-cell');
							if (el) rowHeights[posidx] = el.offsetHeight;
						});
					}}
				>
					<div class="flex flex-col justify-between w-5 h-full">
						<div class="bg-slate-700 rounded-md" style="height: 3px;" />
					</div>
				</div>
			</div>

			<!-- Data columns -->
			{#each $chunk.columnInfos as col, col_index}
				<div
					class={'cell overflow-hidden ' +
						getCellSelectClasses(
							col.name,
							keyidx,
							posidx,
							primarySelectedCell,
							activeCells,
							selectedCells,
							selectedCols,
							selectedRows,
							editMode
						)}
					on:mousedown={editMode &&
					col.name === primarySelectedCell.column &&
					keyidx === primarySelectedCell.keyidx
						? null
						: (e) => {
								e.preventDefault();
								selectCellMethods.mousedown({
									column: col.name,
									keyidx: keyidx,
									posidx: posidx,
									value: $chunk.getCell(rowi, col.name).data
								})(e);
						  }}
					on:dblclick={startEdit}
					column={col.name}
					{keyidx}
				>
					<div
						class={'relative px-1 ' +
							(col.name === primarySelectedCell.column && keyidx === primarySelectedCell.keyidx
								? editMode
									? 'z-10 bg-white border-double border-2 border-violet-600 outline-2 outline-offset-0 outline-violet-300 w-fit '
									: '-mt-px -ml-px '
								: '') +
							(wrapping === 'wrap' ? 'whitespace-pre-wrap ' : 'whitespace-nowrap')}
						style={`min-width: ${columnWidths[col_index] + 2}px;` +
							`min-height: ${rowHeights[posidx] + 2}px;`}
					>
						<Cell
							{...$chunk.getCell(rowi, col.name)}
							editable
							focused={editMode &&
								col.name === primarySelectedCell.column &&
								keyidx === primarySelectedCell.keyidx}
							on:edit={(e) => (editValue = e.detail.value)}
						/>
					</div>
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
					<div class="text-violet-600 font-mono text-sm ">1 row selected</div>
				{:else}
					<CheckAll class="text-violet-600" />
					<div class="text-violet-600 font-mono text-sm ">{selectedRows.length} rows selected</div>
				{/if}
			{/if}
		</div>

		<div class="px-2 flex space-x-2 items-center">
			<div>Wrapping:</div>
			<!-- <label>
				<input type="radio" bind:group={wrapping} name="wrapping" value={'overflow'} />
				Overflow
			</label> -->
			<label>
				<input
					type="radio"
					bind:group={wrapping}
					name="wrapping"
					value={'wrap'}
					on:input={() =>
						setTimeout(() => {
							setColumnWidths();
							setRowHeights();
						}, 100)}
				/>
				Wrap
			</label>
			<label>
				<input type="radio" bind:group={wrapping} name="wrapping" value={'clip'} />
				Clip
			</label>
		</div>

		<Pagination bind:page bind:perPage totalItems={$schema.nrows} dropdownPlacement={'top'} />
	</div>
</div>

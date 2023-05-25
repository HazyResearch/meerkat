<script lang="ts">
	import { get } from 'svelte/store';
	import { getContext, onDestroy } from 'svelte/internal';
	import type { CellInfo } from '$lib/utils/dataframe';

	export let data: any;
	export let editable: boolean = false;
	let editablePrev = editable;
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');

	const cellInfo: CellInfo = getContext('cellInfo');
	// Build up style if it exists in the context
	let style: string = '';
	if (cellInfo) {
		if (cellInfo.style) {
			if (cellInfo.style.minWidth) {
				const unsubscribe = cellInfo.style.minWidth.subscribe((value: number) => {
					style = style.replace(/min-width: \d+px;/, '');
					style += `min-width: ${value}px;`;
				});
				onDestroy(unsubscribe);
			}
			if (cellInfo.style.minHeight) {
				const unsubscribe = cellInfo.style.minHeight.subscribe((value: number) => {
					style = style.replace(/min-height: \d+px;/, '');
					style += `min-height: ${value}px;`;
				});
				onDestroy(unsubscribe);
			}
		}
	}

	let editableCell: HTMLDivElement;

	$: setFocus(editable);

	function setFocus(focus: boolean) {
		if (editablePrev === focus) return;
		editablePrev = focus;

		if (editableCell) {
			if (!focus) {
				editableCell.blur();
				return;
			}

			editableCell.focus();

			// Set the cursor to the end of the div. From
			// https://stackoverflow.com/a/3866442. Supported on Firefox,
			// Chrome, Opera, Safari, IE 9+
			let range = document.createRange();
			range.selectNodeContents(editableCell);
			range.collapse(false);

			let selection = window.getSelection();
			if (selection) {
				selection.removeAllRanges();
				selection.addRange(range);
			}
		}
	}
</script>

<div class="relative">
	<div
		class={classes +
			' px-1 outline-none whitespace-nowrap ' +
			(editable
				? 'bg-white border-double border-2 border-violet-600 outline-2 outline-offset-0 outline-violet-300 w-fit z-50'
				: '')}
		{style}
		contenteditable="true"
		bind:innerHTML={data}
		bind:this={editableCell}
		on:input={() => cellEdit(data)}
	/>
</div>

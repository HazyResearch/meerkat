<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let editable: boolean = false;
	let editablePrev = editable;
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');

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

<div
	class={classes + 'outline-none'}
	contenteditable="true"
	bind:innerHTML={data}
	bind:this={editableCell}
	on:input={() => cellEdit(data)}
/>

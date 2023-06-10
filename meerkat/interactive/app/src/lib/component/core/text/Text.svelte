<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let editable: boolean = false;
	export let focused: boolean = false;
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');

	let editableCell: HTMLDivElement;

	$: setFocus(focused);

	function setFocus(focus: boolean) {
		if (!editableCell) return;

		if (!focus) {
			editableCell.blur();
		} else {
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

{#if editable}
	<!-- TODO: binding innerHTML to data doesn't seem to work when trying to input a newline -->
	<div
		class={classes + ' outline-none'}
		contenteditable="true"
		bind:innerHTML={data}
		bind:this={editableCell}
		on:input={() => cellEdit(data)}
	/>
{:else}
	<div class={classes}>
		{data}
	</div>
{/if}

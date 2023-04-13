<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let editable: boolean = false;
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');

	function handleKeydown(event) {
		// Check if Shift-Enter (Mac)
		const isCmdOrCtrl = event.shiftKey;
		if (event.key === 'Enter' && !isCmdOrCtrl) {
			// Prevent the default behavior (e.g., line break)
			event.preventDefault();
			data = event.target.textContent;
			cellEdit(data);
			event.target.blur();
		}
	}
</script>

{#if editable}
	<div
		class={classes + " whitespace-pre-line"}
		contenteditable="true"
		on:keydown={handleKeydown}
		
	>
		{data}
	</div>
{:else}
	<div class={classes}>
		{data}
	</div>
{/if}

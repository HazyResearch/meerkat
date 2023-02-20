<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let editable: boolean = false;
	export let classes: string = '';
	console.log(classes)

	const cellEdit: CallableFunction = getContext('cellEdit');
</script>



{#if editable}
	<input
		class="input input-bordered grow h-7 px-3 rounded-md shadow-md"
		on:change={() => {cellEdit(data);}}
		bind:value={data}
	/>
{:else}
	<div class={classes}>
		{classes}
		{data}
	</div>

{/if}

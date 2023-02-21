<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let editable: boolean = false;
	export let classes: string = '';

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
		{data}
	</div>

{/if}

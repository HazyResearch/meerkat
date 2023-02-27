<script lang="ts">
	import { getContext } from 'svelte/internal';

	export let data: any;
	export let editable: boolean = false;
	export let classes: string = '';

	const cellEdit: CallableFunction = getContext('cellEdit');
</script>



{#if editable}
	<input
		class="input w-full"
		on:change={() => {cellEdit(data);}}
		bind:value={data}
	/>
{:else}
	<div class={classes}>
		{data}
	</div>

{/if}

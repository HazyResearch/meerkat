<script lang="ts">
	import { Dropdown, DropdownItem } from 'flowbite-svelte';
	export let open: boolean = false;
	export let options: Array<string> = [];
</script>

<div>
	<button
		on:click={() => {
			open = !open;
		}}
	>
		<slot name="button" />
	</button>
	<Dropdown {open}>
		{#each options as option}
			<DropdownItem
				on:click={() => {
					open = false;
				}}>{option}</DropdownItem
			>
		{/each} 
	</Dropdown>
</div>

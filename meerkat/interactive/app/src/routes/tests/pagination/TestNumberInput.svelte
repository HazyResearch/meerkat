<script lang="ts">
	import NumberInput from '$lib/components/pagination/NumberInput.svelte';
	// import { CloudDownload } from 'carbon-icons-svelte';

	let load_more_pages: number = 1;
</script>

<li class="inline-flex">
	<NumberInput bind:value={load_more_pages} min={1} hint_text={'Load pages'} />
	<button
		on:click={() => {}}
		class="inline-flex py-2 px-3 border border-solid dark:bg-gray-800 dark:border-gray-400 dark:hover:bg-gray-700"
	>
		<CloudDownload size={24} />
	</button>
</li>

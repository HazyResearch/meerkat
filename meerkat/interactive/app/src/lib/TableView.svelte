<script lang="ts">
	import Pagination from './components/pagination/Pagination.svelte';
	import Table from '$lib/components/table/Table.svelte';


	export let columns: Array<string> = [];
	export let types: Array<string> = [];
	export let rows: Array<any> = [];
	export let per_page: number = 10;
	export let loader = async (start: number, end: number) => {};

	export let nrows: number = rows.length;
	let page: number = 0;
</script>

<style>
	.table-view {
		@apply h-full overflow-hidden;
		@apply dark:bg-slate-700;
	}
</style>

<div class="table-view">

	<div class="overflow-y-auto overflow-x-hidden h-[700px]">
		<Table bind:columns={columns} bind:types={types} bind:rows={rows}/>
	</div>
	<div class="z-10 top-0 m-2 h-20">
		<Pagination
			bind:page
			bind:per_page
			loaded_items={nrows}
			total_items={nrows}
			loader={loader}
		/>
	</div>
</div>

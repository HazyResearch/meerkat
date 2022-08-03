<script lang="ts">
	import Pagination from './components/pagination/Pagination.svelte';
	import Table from '$lib/components/table/Table.svelte';
	import { api_url } from '../routes/network/stores';
	import { get, post } from '$lib/utils/requests';


	// import Item from './Item.svelte';

	let columns: Array<string> = [];
	let types: Array<string> = [];
	let rows: Array<any> = [];
	let indices: Array<number> = [];

	export let per_page: number = 10;

	export let nrows: number = rows.length;
	export let dp: string = '';
	// let npages: number = Math.ceil(nrows / per_page);
	let page: number = 0;
	// let sorted_by: { col_index: number; ascending: boolean } = { col_index: -1, ascending: true };

	let loader = async (start: number, end: number) => {
		let data_promise = await post(`${$api_url}/dp/${dp}/rows`, { start: start, end: end });

		columns = data_promise.column_info.map((col: any) => col.name);
		rows = data_promise.rows;
		types = data_promise.column_info.map((col: any) => col.type);
		indices = data_promise.indices;
	};
	let data_promise = loader(0, 10);

	// let sort = (col_index: number) => {
	// 	let type = types[col_index];
	// 	let ascending = sorted_by['col_index'] === col_index ? !sorted_by['ascending'] : true;
	// 	sorted_by = { col_index: col_index, ascending: ascending };
	// 	let sorted_rows = _.sortBy(rows, (row: any) => {
	// 		let value = row[col_index];
	// 		if (type === 'number') {
	// 			return value;
	// 		} else {
	// 			return value.toLowerCase();
	// 		}
	// 	});
	// 	if (!ascending) {
	// 		sorted_rows = sorted_rows.reverse();
	// 	}
	// 	rows = sorted_rows;
	// };
</script>

{#await data_promise}
	<div>Loading data...</div>
{:then data}
	<div class="table-view">
		<div class="overflow-y-auto overflow-x-hidden h-[700px]">
			<Table bind:columns bind:types bind:rows />
		</div>
		<div class="z-10 top-0 m-2 h-20">
			<Pagination bind:page bind:per_page loaded_items={nrows} total_items={nrows} {loader} />
		</div>
	</div>
{/await}

<style>
	.table-view {
		@apply h-full overflow-hidden;
		@apply dark:bg-slate-700;
	}
</style>

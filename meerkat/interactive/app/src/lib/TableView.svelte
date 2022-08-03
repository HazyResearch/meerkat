<script lang="ts">
	// import { SortDown, SortUp } from 'svelte-bootstrap-icons';
	// import _, { forEach, zip } from 'underscore';
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

<!-- <div class="bg-slate-800 p-2">
    
    <div class="fixed z-20 top-1 left-0 right-0 m-2">
        <Pagination
            bind:page
            bind:per_page
            loaded_items={nrows}
            total_items={nrows}
            loader={async (start, end) => {}}
        />
    </div>
    
    <div class="m-3 overflow-auto relative top-20">
		<table class="table-fixed relative w-full text-sm text-left text-gray-500 dark:text-gray-400">
			<thead class="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
				<tr>
					{#each columns as column, col_index}
						<th class="py-2 px-4 resize-x [overflow:hidden]">
							<div class="flex items-center">
								<button
									on:click={() => {
										sort(col_index);
									}}
								>
                                    {column}
									{#if sorted_by.col_index === col_index}
										{#if sorted_by.ascending}
											<SortDown/>
										{:else}
											<SortUp/>
										{/if}
									{/if}
								</button>
							</div>
						</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each rows.slice(page * per_page, (page + 1) * per_page) as row}
					<tr
						class="bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600"
					>
						{#each zip(row, types) as [value, type]}
							<td class="py-2 px-4 overflow-auto break-words">
								<Item data={value} />
							</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</div> -->

<style>
	.table-view {
		@apply h-full overflow-hidden;
		@apply dark:bg-slate-700;
	}
</style>

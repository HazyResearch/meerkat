<script lang="ts">
	import Pagination from '$lib/components/pagination/Pagination.svelte';
	import Gallery from './Cards.svelte';
	import GallerySlider from './GallerySlider.svelte';
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { BarLoader } from 'svelte-loading-spinners';

	const { get_schema, get_rows, edit } = getContext('Interface');

	export let dp: Writable;
	export let main_column: Writable<string>;
	export let tag_columns: Writable<Array<string>>;
	export let edit_target: Any;

	export let page: number = 0;
	export let per_page: number = 100;

	export let cell_size: number = 12;

	$: schema_promise = $get_schema($dp.box_id);
	$: rows_promise = $get_rows($dp.box_id, page * per_page, (page + 1) * per_page);

	async function handle_edit(event: any) {
		let { pivot, pivot_id_column, id_column } = edit_target;
		let rows = await rows_promise;

		let { row, column, value } = event.detail;
		let row_id_column_index = rows.column_infos.findIndex((c) => c.name === id_column);
		let row_index = rows.indices.indexOf(row);
		let row_id = rows.rows[row_index][row_id_column_index];

		$edit(pivot.box_id, value, column, row_id, pivot_id_column);
	}
</script>

<div class="flex-1 rounded-lg overflow-hidden bg-slate-50">
	{#await schema_promise}
		<div class="flex justify-center items-center h-full">
			<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
		</div>
	{:then schema}
		<div class="h-full grid grid-rows-[auto_1fr] relative">
			<div class="grid grid-cols-3 h-12 z-10 rounded-t-lg drop-shadow-xl bg-slate-100">
				<div class="font-semibold self-center">
					
				</div>
				<span class="font-bold text-xl text-slate-600 self-center justify-self-center"> Gallery </span>
				<span class="font-semibold self-center justify-self-end"> 
					<GallerySlider bind:size={cell_size} />	 
				</span>
			</div>
			<div class="h-full overflow-y-scroll">
				{#await rows_promise}
					<div class="h-full flex items-center justify-center">
						<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
					</div>
				{:then rows}
					<Gallery {schema} {rows} main_column={$main_column} tag_columns={$tag_columns} bind:cell_size={cell_size} />
				{:catch error}
					{error}
				{/await}
			</div>

			<div class="absolute z-10 bottom-0 w-full m-0 px-14 pb-3">
				<Pagination
					bind:page
					bind:per_page
					loaded_items={schema.nrows}
					total_items={schema.nrows}
				/>
			</div>
		</div>
	{:catch error}
		{error}
	{/await}
</div>

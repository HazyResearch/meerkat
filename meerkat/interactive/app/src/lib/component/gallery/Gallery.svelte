<script lang="ts">
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import Cards from './Cards.svelte';
	import GallerySlider from './GallerySlider.svelte';
	import { getContext, setContext } from 'svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import Selected from './Selected.svelte';

	import { openModal } from 'svelte-modals';
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import type { DataFrameRef } from '$lib/api/dataframe';

	const { get_schema, get_rows } = getContext('Meerkat');

	export let df: DataFrameRef;
	export let main_column: string;
	export let tag_columns: Any; // Writable<Array<string>>;
	export let selected: Array<string>;

	export let page: number = 0;
	export let per_page: number = 20;

	export let cell_size: number = 24;

	$: schema_promise = get_schema(df.ref_id);

	setContext('open_row_modal', (posidx: number) => {
		openModal(RowModal, {
			df: df,
			posidx: posidx,
			main_column: main_column,
		});
	});

	// create an array with the main_column and the tag_columns
	$: df_slice_promise = get_rows(
		df.ref_id,
		page * per_page,
		(page + 1) * per_page
		// TODO (Sabri): we should limit the columns only to the main_column and the
		// tag_columns and primary_key as described below
		// null,
		// [$main_column, primary_key].concat($tag_columns)
	);

	// async function handle_edit(event: any) {
	// 	let { pivot, pivot_id_column, id_column } = edit_target;
	// 	let rows = await rows_promise;

	// 	let { row, column, value } = event.detail;
	// 	let row_id_column_index = rows.column_infos.findIndex((c) => c.name === id_column);
	// 	let row_index = rows.indices.indexOf(row);
	// 	let row_id = rows.rows[row_index][row_id_column_index];

	// 	edit(pivot.ref_id, value, column, row_id, pivot_id_column);
	// }
</script>

<div class="flex-1 rounded-lg overflow-hidden bg-slate-50">
	{#await schema_promise}
		<div class="flex justify-center items-center h-full">
			<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
		</div>
	{:then schema}
		<div class="h-full grid grid-rows-[auto_1fr] relative">
			<div class="grid grid-cols-3 h-12 z-10 rounded-t-lg drop-shadow-xl bg-slate-100">
				<div class="font-semibold self-center px-10 flex space-x-2">
					{#if selected.length > 0}
						<Selected />
						<div class="text-violet-600">{selected.length}</div>
					{/if}
				</div>
				<span class="font-bold text-xl text-slate-600 self-center justify-self-center">
					Examples
				</span>
				<span class="font-semibold self-center justify-self-end">
					<GallerySlider bind:size={cell_size} />
				</span>
			</div>
			<div class="h-full overflow-y-scroll">
				{#await df_slice_promise}
					<div class="h-full flex items-center justify-center">
						<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
					</div>
				{:then df_slice}
					<Cards
						{schema}
						rows={df_slice}
						{main_column}
						{tag_columns}
						bind:cell_size
						bind:selected
					/>
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

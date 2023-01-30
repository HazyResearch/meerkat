<script lang="ts">
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import Cards from './Cards.svelte';
	import GallerySlider from './GallerySlider.svelte';
	import { getContext, setContext } from 'svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import Selected from './Selected.svelte';
	import { openModal } from 'svelte-modals';
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import { Dropdown, DropdownItem } from 'flowbite-svelte';
	import type { DataFrameRef } from '$lib/api/dataframe';

	const { fetch_schema, fetch_chunk } = getContext('Meerkat');

	export let df: DataFrameRef;
	export let main_column: string;
	export let tag_columns: Array<string>;
	export let selected: Array<string>;

	export let page: number = 0;
	export let per_page: number = 20;
	export let cell_size: number = 24;

	$: schema_promise = fetch_schema({
		df: df,
		variants: ['small']
	});

	setContext('open_row_modal', (posidx: number) => {
		openModal(RowModal, {
			df: df,
			posidx: posidx,
			main_column: main_column
		});
	});

	$: chunk_promise = fetch_chunk({
		df: df,
		start: page * per_page,
		end: (page + 1) * per_page,
		columns: [main_column, ...tag_columns],
		variants: ['small']
	});

	let dropdown_open: boolean = false;
</script>

<div class="flex-1 rounded-lg overflow-hidden bg-slate-50 h-full">
	{#await schema_promise}
		<div class="flex justify-center items-center h-full">
			<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
		</div>
	{:then schema}
		<div class="h-full grid grid-rows-[auto_1fr] relative">
			<div class="grid grid-cols-3 h-12 z-10 rounded-t-lg drop-shadow-xl bg-slate-100 px-5">
				<!-- Left header section -->
				<div class="flex justify-self-start items-center">
					<span class="font-semibold">
						<GallerySlider bind:size={cell_size} />
					</span>
					<div class="font-semibold self-center px-10 flex space-x-2">
						{#if selected.length > 0}
							<Selected />
							<div class="text-violet-600">{selected.length}</div>
						{/if}
					</div>
				</div>

				<!-- Middle header section -->
				<div class="self-center justify-self-center">
					<button
						class="font-bold font-mono text-xl text-slate-600 self-center justify-self-center"
						on:click={() => {
							dropdown_open = !dropdown_open;
						}}
					>
						{main_column}
					</button>
					<Dropdown open={dropdown_open} class="w-fit">
						{#each schema.columns as col}
							<DropdownItem
								on:click={() => {
									main_column = col.name;
									dropdown_open = false;
								}}
							>
								<div class="text-slate-600 font-mono">
									<span class="font-bold">{col.name}</span>
								</div>
							</DropdownItem>
						{/each}
					</Dropdown>
				</div>

				<!-- Right header section -->
				<div class="flex self-center justify-self-end items-center">
					<Pagination bind:page bind:per_page total_items={schema.nrows} />
				</div>
			</div>
			<div class="h-full overflow-y-scroll">
				{#await chunk_promise}
					<div class="h-full flex items-center justify-center">
						<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
					</div>
				{:then chunk}
					<Cards {schema} {chunk} {main_column} {tag_columns} bind:cell_size bind:selected />
				{:catch error}
					{error}
				{/await}
			</div>
		</div>
	{:catch error}
		{error}
	{/await}
</div>

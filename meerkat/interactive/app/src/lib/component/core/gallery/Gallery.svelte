<script lang="ts">
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import { Dropdown, DropdownItem } from 'flowbite-svelte';
	import { setContext, getContext } from 'svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import { openModal } from 'svelte-modals';
	import Cards from './Cards.svelte';
	import GallerySlider from './GallerySlider.svelte';
	import Selected from './Selected.svelte';

	export let df: DataFrameRef;
	export let mainColumn: string;
	export let tagColumns: Array<string>;
	export let selected: Array<string>;

	export let page: number = 0;
	export let perPage: number = 20;
	export let cellSize: number = 24;

	export let allowSelection: boolean = false;

	$: schemaPromise = fetchSchema({
		df: df,
		formatter: 'small'
	});

	setContext('open_row_modal', (posidx: number) => {
		openModal(RowModal, {
			df: df,
			posidx: posidx,
			mainColumn: mainColumn
		});
	});

	$: chunkPromise = fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		columns: [mainColumn, ...tagColumns],
		formatter: {
			[mainColumn]: 'gallery',
			...tagColumns.reduce((acc, col) => ({ ...acc, [col]: 'tag' }), {})
		}
	});

	let dropdownOpen: boolean = false;
</script>

<div class="flex-1 rounded-lg overflow-hidden bg-slate-50 h-full shadow-md">
	{#await schemaPromise}
		<div class="flex justify-center items-center h-full">
			<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
		</div>
	{:then schema}
		<div class="h-full grid grid-rows-[auto_1fr] relative">
			<div class="grid grid-cols-3 h-12 z-10 rounded-t-lg drop-shadow-xl bg-slate-100 px-5">
				<!-- Left header section -->
				<div class="flex justify-self-start items-center">
					<span class="font-semibold">
						<GallerySlider bind:size={cellSize} />
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
							dropdownOpen = !dropdownOpen;
						}}
					>
						{mainColumn}
					</button>
					<Dropdown open={dropdownOpen} class="w-fit">
						{#each schema.columns as col}
							<DropdownItem
								on:click={() => {
									mainColumn = col.name;
									dropdownOpen = false;
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
					<Pagination bind:page bind:perPage totalItems={schema.nrows} />
				</div>
			</div>
			<div class="h-full overflow-y-scroll">
				{#await chunkPromise}
					<div class="h-full flex items-center justify-center">
						<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
					</div>
				{:then chunk}
					<Cards {chunk} {mainColumn} {tagColumns} {allowSelection} bind:cellSize bind:selected />
				{:catch error}
					{error}
				{/await}
			</div>
		</div>
	{:catch error}
		{error}
	{/await}
</div>

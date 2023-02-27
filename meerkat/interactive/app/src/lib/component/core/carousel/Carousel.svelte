<script lang="ts">
	import Cell from '$lib/shared/cell/Cell.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema } from '$lib/utils/api';
	import { DataFrameChunk, type DataFrameRef, type DataFrameSchema } from '$lib/utils/dataframe';
	import { writable, type Writable } from 'svelte/store';

	export let df: DataFrameRef;
	export let mainColumn: string;

	let page: number = 0;
	let perPage: number = 4;

	let schema: Writable<DataFrameSchema> = writable({
		columns: [],
		primaryKey: 'pkey',
		nrows: 0,
		id: ''
	});
	let chunk: Writable<DataFrameChunk> = writable(new DataFrameChunk([], [], [], 0, 'pkey'));

	$: fetchSchema({
		df: df,
		formatter: 'small'
	}).then((newSchema) => {
		schema.set(newSchema);
	});

	$: fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
		columns: [mainColumn],
		formatter: {
			[mainColumn]: 'gallery'
		}
	}).then((newChunk) => {
		chunk.set(newChunk);
	});
</script>

<div class="flex-1 rounded-lg overflow-hidden bg-slate-50 h-full shadow-md py-1">
	<div class="h-full flex flex-col relative">
        <div class="flex self-center justify-self-end items-center">
            <Pagination bind:page bind:perPage totalItems={$schema.nrows} allowSetPerPage={false} />
        </div>
		<div class="h-full overflow-y-scroll">
			<div class="h-12 flex flex-shrink justify-center gap-4">
				{#each $chunk.keyidxs as keyidx, i}
					<div class="flex-shrink">
						<Cell {...$chunk.getCell(i, mainColumn)} />
					</div>
				{/each}
			</div>
		</div>
	</div>
</div>

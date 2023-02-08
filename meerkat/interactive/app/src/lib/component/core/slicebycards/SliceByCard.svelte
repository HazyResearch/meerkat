<script lang="ts">
	// import Description from './Description.svelte';
	import type { DataFrameRows, DataFrameSchema } from '$lib/api/dataframe';
	import LoadButton from '$lib/shared/common/LoadButton.svelte';
	import Pill from '$lib/shared/common/Pill.svelte';
	import type { SliceByBox } from '$lib/utils/types';
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import RowCard from './RowCard.svelte';

	const { get_sliceby_rows } = getContext('Meerkat');

	export let sliceby: Writable<SliceByBox>;
	export let slice_key: string | number;
	export let schema: DataFrameSchema;
	export let main_column: string;
	export let tag_columns: Array<string>;
	export let aggregations_promise: any;

	let columns = schema.columns.map((col: any) => col.name);
	let tag_indices: Array<number> = tag_columns.map((tag) => columns.indexOf(tag));
	let main_index: number = columns.indexOf(main_column);
	let main_component = schema.columns[main_index].cell_component;
	let main_props = schema.columns[main_index].cell_props;

	let page: number = 0;
	const per_page: number = 25;

	let rows: DataFrameRows = {
		rows: [],
		indices: [],
		full_length: 0
	};

	let load_status = 'waiting';
	let load_rows = async () => {
		load_status = 'loading';
		let new_rows = await get_sliceby_rows(
			$sliceby.ref_id,
			slice_key,
			page * per_page,
			(page + 1) * per_page
		);
		rows.indices = rows.indices.concat(new_rows.indices);
		rows.rows = rows.rows.concat(new_rows.rows);
		page = page + 1;
		load_status = 'waiting';
	};
	load_rows();

	let get_type;
</script>

<div class="pt-2 pl-6 pb-2 h-fit mx-auto bg-white rounded-xl shadow-lg overflow-x-hidden ml-4">
	<div class="h-10 mb-2 flex space-x-6 items-center">
		<div class="flex space-x-3 whitespace-nowrap text-xl text-black">
			<div>Slice</div>
			<div class="font-bold">{slice_key}</div>
		</div>
		<!-- <div class="mx-auto flex h-full items-center space-x-4 overflow-x-scroll no-scrollbar ml-4">
			{#each descriptions as description}
				<Description score={description.score} description={description.description} />
			{/each}
		</div> -->
	</div>
	<div class="flex h-40">
		<!-- <Stats {stats} /> -->
		<div class="flex-row">
			{#await aggregations_promise}
				Loading aggregations...
			{:then aggregations}
				{#each Object.entries(aggregations) as [name, aggregation]}
					<Pill layout="wide-content" header={name} content={aggregation.df[slice_key]} />
				{/each}
			{:catch error}
				{error}
			{/await}
		</div>

		<div class="flex overflow-x-scroll mx-2 items-center space-x-4">
			{#if rows.rows.length < 1}
				<LoadButton status="loading" on:load={load_rows} />
			{:else}
				{#each rows.rows as row, index}
					<RowCard
						id={index.toString()}
						main={{
							data: row[main_index],
							cellComponent: main_component,
							cellProps: {
								...main_props
							}
						}}
						tags={tag_indices.map((tag_index) => ({
							data: row[tag_index],
							cell_component: schema.columns[tag_index].cell_component,
							cell_props: {
								...schema.columns[tag_index].cell_props
							}
						}))}
						layout="gimages"
					/>
				{/each}
				<LoadButton status={'waiting'} on:load={load_rows} />
			{/if}
		</div>
	</div>
</div>

<style>
	.no-scrollbar::-webkit-scrollbar {
		display: none;
	}
	.no-scrollbar {
		-ms-overflow-style: none; /* IE and Edge */
		scrollbar-width: none; /* Firefox */
	}
</style>

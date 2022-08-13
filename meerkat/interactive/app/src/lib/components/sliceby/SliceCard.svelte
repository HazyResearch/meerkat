<script lang="ts">
	// import Description from './Description.svelte';
	import type { DataPanelRows, ColumnInfo, DataPanelSchema } from '$lib/api/datapanel';
	import { api_url } from '../../../routes/network/stores';
	import { get_rows } from '$lib/api/sliceby';
    import type {SliceKey} from '$lib/api/sliceby';
	import RowCard from '../row/RowCard.svelte';
	import LoadButton from '../common/LoadButton.svelte';
	// import Matrix, { type MatrixType } from './matrix/Matrix.svelte';
	// import Stats, { type StatsType } from './Stats.svelte';

	export let sliceby_id: string;
	export let datapanel_id: string;
	export let slice_key: SliceKey;
	export let schema: DataPanelSchema;
	export let main_column: string;
	export let tag_columns: Array<string>;

	let columns = schema.columns.map((col: any) => col.name);
	let tag_indices: Array<number> = tag_columns.map((tag) => columns.indexOf(tag));
	let main_index: number = columns.indexOf(main_column);
	let main_component = schema.columns[main_index].cell_component;
	let main_props = schema.columns[main_index].cell_props;
	let tag_component = tag_indices.map((tag_index) => schema.columns[tag_index].cell_component);
	let tag_props = tag_indices.map((tag_index) => schema.columns[tag_index].cell_props);

	let page: number = 0;
	const per_page: number = 25;

	let rows: DataPanelRows = {
		rows: [],
		indices: [],
		full_length: 0
	};

	let load_status = 'waiting';
	let load_rows = async () => {
		load_status = 'loading';
		let new_rows = await get_rows(
			$api_url,
			sliceby_id,
			slice_key,
			page * per_page,
			(page + 1) * per_page
		);
		rows.indices = rows.indices.concat(new_rows.indices);
		rows.rows = rows.rows.concat(new_rows.rows);
		page = page + 1;
		load_status = 'waiting';
	};
	let load_promise = load_rows();
</script>

<div class="pt-2 px-6 pb-2 h-fit mx-auto bg-white rounded-xl shadow-lg overflow-x-hidden ml-4">
	<div class="h-10 mb-2 flex space-x-6 items-center">
		<div class="whitespace-nowrap text-xl font-medium text-black">
			{slice_key}
		</div>
		<!-- <div class="mx-auto flex h-full items-center space-x-4 overflow-x-scroll no-scrollbar ml-4">
			{#each descriptions as description}
				<Description score={description.score} description={description.description} />
			{/each}
		</div> -->
	</div>
	<div class="flex h-40">
		<!-- <Stats {stats} /> -->

		<!-- {#if plot.PLOT_TYPE === 'matrix' && plot.matrix}
			<Matrix {...plot.matrix} />
		{:else if plot.PLOT_TYPE === 'plotly'}
			{plot.html}
		{/if} -->

		<div class="flex overflow-x-scroll no-scrollbar ml-4 items-center space-x-4">
			{#if rows.rows.length < 1}
				<LoadButton status="loading" on:load={load_rows} />
			{:else}
				{#each rows.rows as row, index}
					<RowCard
						id={index.toString()}
						main={{
							data: row[main_index],
							cell_component: main_component,
							cell_props: {
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
				<LoadButton status={load_status} on:load={load_rows} />
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

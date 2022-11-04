<script lang="ts">
	import type { RefreshCallback } from '$lib/api/callbacks';
	import {
		filter,
		FilterCriterion,
		get_rows,
		get_schema,
		match,
		MatchCriterion,
		sort,
		type DataFrameSchema
	} from '$lib/api/dataframe';
	import Tab from '$lib/shared/header/Tab.svelte';
	import Tabs from '$lib/shared/header/Tabs.svelte';
	import MatchHeader from '$lib/shared/match_header/MatchHeader.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import PlotHeader from '$lib/shared/plot_header/PlotHeader.svelte';
	import Table from '$lib/shared/table/Table.svelte';
	import { api_url } from '../routes/network/stores';
	import DummyBlock from './shared/blocks/DummyBlock.svelte';
	import Everything from './shared/blocks/Everything.svelte';
	import FilterHeader from './shared/filter_header/FilterHeader.svelte';
	import Gallery from './shared/gallery/Cards.svelte';
	import CategoryGenerator from './shared/lm/CategoryGenerator.svelte';

	export let dataframe_id: string;
	export let nrows: number = 0;
	export let page: number = 0;
	export let per_page: number = 100;

	const base_dataframe_id: string = dataframe_id;
	let filter_criteria: Array<FilterCriterion> = [];
	let match_criterion: MatchCriterion = new MatchCriterion('', ''); // TODO: clean this up to be null.

	const refresh: RefreshCallback = async () => {
		let curr_dataframe_id = base_dataframe_id;
		console.log(match_criterion);
		console.log(filter_criteria);

		// Run operations: match -> sort -> filter.
		// match is performed first to ensure columns are added to the base dataframe.
		// TODO (arjundd): Figure out a way to put sorting after filtering for time efficiency.

		let run_match: boolean = match_criterion.query !== '' && match_criterion.column !== '';
		let run_filter: boolean = filter_criteria.length > 0;
		let sort_by_column: string = '';

		console.log('df id start: ', curr_dataframe_id);
		console.log('run_match: ', run_match);

		type PromiseLambda = {
			(): Promise<any>;
		};

		let op_promises: Array<PromiseLambda> = [];
		let op_names: Array<string> = [];
		// Match.
		if (run_match) {
			// Push a promise that matches --> sorts --> updates dataframe_id
			op_promises.push(async () => {
				// Run match
				let schema_after_match = await match($api_url, curr_dataframe_id, match_criterion);

				const previous_df_id = curr_dataframe_id;
				curr_dataframe_id = schema_after_match.id;
				sort_by_column = schema_after_match.columns[0].name;
				console.log('match: ', previous_df_id, ' -> ', curr_dataframe_id, sort_by_column);

				let schema_after_sort = await sort($api_url, curr_dataframe_id, sort_by_column);

				console.log('sort: ', curr_dataframe_id, ' -> ', schema_after_sort.id);
				curr_dataframe_id = schema_after_sort.id;
			});

			op_names.push('match');
		}
		// Filter.
		if (run_filter) {
			op_promises.push(() =>
				filter($api_url, curr_dataframe_id, filter_criteria).then((schema: DataFrameSchema) => {
					let previous_df_id = curr_dataframe_id;
					curr_dataframe_id = schema.id;
					console.log('filter: ', previous_df_id, ' -> ', curr_dataframe_id);
				})
			);
			op_names.push('filter');
		}
		// Sort.
		// if (run_match) {
		// 	op_promises.push(sort(
		// 		$api_url, curr_dataframe_id, sort_by_column
		// 	).then((schema: DataFrameSchema) => {
		// 		console.log("sort, sort by col: ", sort_by_column)
		// 		curr_dataframe_id = schema.id;
		// 		console.log("sort: ", curr_dataframe_id);
		// 	}));
		// }
		if (op_promises.length == 0) {
			let promise = new Promise(() => {
				dataframe_id = base_dataframe_id;
			});
			return promise;
		}

		console.log('op promise: ', op_promises[0]);
		console.log('Ops: ', op_promises);
		console.log('Op names: ', op_names);
		let op_promise: Promise<any> = op_promises[0]();
		for (let i = 1; i < op_promises.length; i++) {
			op_promise = op_promise.then(op_promises[i]);
		}
		op_promise.then(() => {
			dataframe_id = curr_dataframe_id;
		});
		return op_promise;
	};

	$: schema_promise = get_schema($api_url, dataframe_id).then((schema) => {
		schema.columns = schema.columns.filter((column: any) => {
			return !column.name.startsWith('__');
		});
		return schema;
	});
	$: rows_promise = get_rows($api_url, dataframe_id, page * per_page, (page + 1) * per_page);

	let on_selection_change = async (event: CustomEvent) => {
		/* Triggered when brush selection on scatter plot changes. */
		// Update rows_promise: call get_rows again with the new selection (indices)
		rows_promise = get_rows(
			$api_url,
			base_dataframe_id,
			undefined,
			undefined,
			Array.from(event.detail.selected_points)
		);
		console.log('On Selection Change');
		console.log(event.detail.selected_points);
	};

	let toggle_button: boolean = false;
	$: active_view = toggle_button ? 'gallery' : 'table';
</script>

<Tabs bind:toggle_button>
	<Tab label="Match" id="match">
		<MatchHeader bind:match_criterion {schema_promise} refresh_callback={refresh} />
	</Tab>
	<Tab label="Filter" id="filter">
		<FilterHeader bind:filter_criteria {schema_promise} refresh_callback={refresh} />
	</Tab>
	<Tab label="Info" id="info">
		<CategoryGenerator
			categories={[
				'age',
				'ethnicity',
				'facial expression',
				'resolution',
				'blurry',
				'age',
				'ethnicity',
				'facial expression',
				'resolution',
				'blurry',
				'age',
				'ethnicity',
				'facial expression',
				'resolution',
				'blurry'
			]}
		/>
	</Tab>
	<Tab label="Block" id="block">
		<Everything />
	</Tab>

	<Tab label="Plot" id="plot">
		<PlotHeader
			bind:match_criterion
			dataframe_id={base_dataframe_id}
			{rows_promise}
			{schema_promise}
			refresh_callback={refresh}
			on:selection-change={on_selection_change}
		/>
	</Tab>
</Tabs>

{#if active_view === 'table'}
	<div class="table-view">
		{#await schema_promise}
			<div class="h-full">Loading data...</div>
		{:then schema}
			<div class="overflow-y-auto overflow-x-hidden h-full">
				{#await rows_promise}
					<Table rows={null} {schema} />
				{:then rows}
					<Table {rows} {schema} />
				{/await}
			</div>
		{/await}
		<div class="z-10 top-0 m-0 h-20">
			<Pagination bind:page bind:per_page loaded_items={nrows} total_items={nrows} />
		</div>
	</div>
{:else if active_view === 'gallery'}
	<div class="table-view">
		{#await schema_promise}
			<div class="h-full">Loading gallery...</div>
		{:then schema}
			<div class="overflow-y-auto overflow-x-hidden h-full">
				{#await rows_promise}
					<Gallery {schema} rows={{ rows: [] }} />
				{:then rows}
					<Gallery {schema} {rows} main_column={'image'} tag_columns={['v6_fnmr', 'ethnicity']} />
				{/await}
			</div>
		{/await}
		<div class="z-10 top-0 m-0 h-20">
			<Pagination bind:page bind:per_page loaded_items={nrows} total_items={nrows} />
		</div>
	</div>
{/if}

<style>
	.table-view {
		@apply h-full grid grid-rows-[1fr_auto] overflow-hidden;
		@apply dark:bg-slate-700;
	}
</style>

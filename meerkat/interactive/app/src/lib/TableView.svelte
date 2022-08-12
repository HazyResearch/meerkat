<script lang="ts">
	import { FilterCriterion, get_rows, get_schema, filter, match, sort, type DataPanelSchema, MatchCriterion } from '$lib/api/datapanel';
	import type { RefreshCallback, NoArgCallback } from '$lib/api/callbacks';
	import Table from '$lib/components/table/Table.svelte';
	import { api_url } from '../routes/network/stores';
	import { writable } from 'svelte/store';
	import Pagination from '$lib/components/pagination/Pagination.svelte';
	import Toggle from './components/common/Toggle.svelte';
	import Gallery from './components/gallery/Gallery.svelte';
	import Header from './components/header/Header.svelte';
	import MatchHeader from '$lib/components/match_header/MatchHeader.svelte';
	import Tabs from '$lib/components/header/Tabs.svelte';
	import Tab from '$lib/components/header/Tab.svelte';
	import FilterHeader from './components/filter_header/FilterHeader.svelte';

	export let datapanel_id: string;
	export let nrows: number = 0;
	export let page: number = 0;
	export let per_page: number = 10;

	const base_datapanel_id: string = datapanel_id;
	let filter_criteria: Array<FilterCriterion> = [];
	let match_criterion: MatchCriterion = new MatchCriterion("", "");  // TODO: clean this up to be null.

	const refresh: RefreshCallback = async () => {
		let curr_datapanel_id = base_datapanel_id
		console.log(match_criterion)
		console.log(filter_criteria)

		// Run operations: match -> sort -> filter.
		// match is performed first to ensure columns are added to the base datapanel.
		// TODO (arjundd): Figure out a way to put sorting after filtering for time efficiency.
		let run_match: boolean = (match_criterion.query !== "") && (match_criterion.column !== "");
		let run_filter: boolean = (filter_criteria.length > 0)
		let sort_by_column: string = "";

		console.log("dp id start: ", curr_datapanel_id)
		
		type PromiseLambda = {
			(): Promise<any>;
		}

		let op_promises: Array<PromiseLambda> = [];
		let op_names: Array<string> = [];
		// Match.
		if (run_match) {
			op_promises.push(() => match(
				$api_url,
				curr_datapanel_id,
				match_criterion
			).then((schema: DataPanelSchema) => {
				let previous_dp_id = curr_datapanel_id;
				curr_datapanel_id = schema.id;
				sort_by_column = schema.columns[0].name;
				console.log("match: ", previous_dp_id, " -> ", curr_datapanel_id, sort_by_column);
				return sort(
					$api_url, curr_datapanel_id, sort_by_column
				)
			}).then((schema: DataPanelSchema) => {
				console.log("sort: ", curr_datapanel_id, " -> ", schema.id);
				curr_datapanel_id = schema.id;
			})
			);
			op_names.push("match");
		}
		// Filter.
		if (filter_criteria.length > 0) {
			op_promises.push(() => filter(
				$api_url,
				curr_datapanel_id,
				filter_criteria
			).then((schema: DataPanelSchema) => {
				let previous_dp_id = curr_datapanel_id;
				curr_datapanel_id = schema.id;
				console.log("filter: ", previous_dp_id, " -> ", curr_datapanel_id,);
			}));
			op_names.push("filter");
		}
		// Sort.
		// if (run_match) {
		// 	op_promises.push(sort(
		// 		$api_url, curr_datapanel_id, sort_by_column
		// 	).then((schema: DataPanelSchema) => {
		// 		console.log("sort, sort by col: ", sort_by_column)
		// 		curr_datapanel_id = schema.id;
		// 		console.log("sort: ", curr_datapanel_id);
		// 	}));
		// }
		if (op_promises.length == 0) {
			let promise = new Promise(() => {datapanel_id = base_datapanel_id})
			return promise
		}
		
		console.log("op promise: ", op_promises[0])
		console.log("Ops: ", op_promises)
		console.log("Op names: ", op_names)
		let op_promise: Promise<any> = op_promises[0]();
		for (let i = 1; i < op_promises.length; i++) {
			op_promise = op_promise.then(op_promises[i]);
		}
		op_promise.then(() => {datapanel_id = curr_datapanel_id;})
		return op_promise;
	};

	$: schema_promise = get_schema($api_url, datapanel_id);
	$: rows_promise = get_rows($api_url, datapanel_id, page * per_page, (page + 1) * per_page);

	let toggle_button: boolean = false;
	$: active_view = toggle_button ? 'gallery' : 'table';


</script>

<!-- <div class="inline-flex mb-4">
	<Toggle label_left="Table" label_right="Gallery" bind:checked={toggle_button} />
</div> -->

<Tabs bind:toggle_button>
	<Tab label="Match" id="match">
		<MatchHeader bind:match_criterion={match_criterion} {schema_promise} refresh_callback={refresh} />
	</Tab>
	<Tab label="Filter" id="filter">
		<FilterHeader bind:filter_criteria={filter_criteria} {schema_promise} refresh_callback={refresh} />
	</Tab>
	<Tab label="Info" id="info">second</Tab>
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
					<Gallery {schema} {rows} main_column={'img'} tag_columns={['label_idx', 'split']} />
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

<script lang="ts">
	import type { FilterCriterion } from '$lib/component/core/filter/types';
	import Status from '$lib/shared/common/Status.svelte';
	import type { RefreshCallback } from '$lib/shared/deprecate/callbacks';
	import type { DataFrameSchema } from '$lib/utils/dataframe';
	import Select from 'svelte-select';

	export let schema_promise: Promise<DataFrameSchema>;
	export let refresh_callback: RefreshCallback;
	export let filter_criteria: Array<FilterCriterion>;

	let column: string = '';
	let op: string = '';
	let filter_value: string = '';
	let status: string = 'waiting';

	const empty_items: Array<any> = [];
	let items_promise = schema_promise.then((schema) => {
		return schema.columns.map((column) => {
			return {
				value: column.name,
				label: column.name
			};
		});
	});

	const ops:Array<string> = ["==", ">", "<", ">=", "<="];

	const add_filter = () => {
		if (column.value === '' || op.value === '' || filter_value === '') {
			console.log('empty');
			status = "error";
			return;
		}
		filter_criteria = [...filter_criteria, {
			column: column.value,
			op: op.value,
			value: filter_value
		}];

		let promise = refresh_callback();
		promise.then(() => {
			status = 'success';
		});
	}

	const clear_filter = () => {
		filter_criteria = [];
		let promise = refresh_callback();
		promise.then(() => {
			status = 'success';
		});
	}

	const onKeyPress = (e) => {
		if (e.charCode === 13) add_filter();
		else status = 'waiting';
	};

</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md">
	<div class="form-control w-full">
		<div class="input-group w-full flex items-center">
			<div class="px-3">
				{#await schema_promise}
					<Status status="working" />
				{:then items}					
					<Status status={status} />
				{/await}
			</div>

			<div class="themed pr-2">
				{#await items_promise}
					<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
				{:then items}
					<Select
						id="column"
						placeholder="...a column."
						bind:value={column}
						{items}
						showIndicator={true}
						listPlacement="auto"
					/>
				{/await}
			</div>

			<div class="themed pr-2">
				<Select
					id="op"
					placeholder="...an operation."
					bind:value={op}
					items={ops}
					showIndicator={true}
					listPlacement="auto"
				/>
			</div>

			<div class="themed pr-2">
				<input
					type="text"
					id="filter_value"
					bind:value={filter_value}
					placeholder="...a value"
					on:keypress={onKeyPress}
				/>
			</div>

			<div class="themed pr-2">
				<button
					on:click={add_filter}
					class="inline-flex py-2 px-3 border border-solid dark:bg-gray-800 dark:border-gray-400 dark:hover:bg-gray-700"
				>
				Add Filter
				</button>
			</div>
			<div class="themed pr-2">
				<button
					on:click={clear_filter}
					class="inline-flex py-2 px-3 border border-solid dark:bg-gray-800 dark:border-gray-400 dark:hover:bg-gray-700"
				>
				Clear Filter
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	/* 	
			CSS variables can be used to control theming.
			https://github.com/rob-balfre/svelte-select/blob/master/docs/theming_variables.md
	*/

	.themed {
		--itemPadding: 0.1rem;
		--itemColor: '#7c3aed';
		@apply rounded-md w-40 border-0;
		@apply z-[1000000];
	}
	.list-container {
		z-index: 2000;
		position: relative;
	}
</style>
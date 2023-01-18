<script lang="ts">
	import type { Writable } from 'svelte/store';
	import type { FilterCriterion, DataFrameSchema } from '$lib/api/dataframe';
	import { getContext } from 'svelte';
	import Select from 'svelte-select';
	import { toast } from '@zerodevx/svelte-toast';
	// Bootstrap icons
	import XCircle from 'svelte-bootstrap-icons/lib/XCircle.svelte';
	import X from 'svelte-bootstrap-icons/lib/X.svelte';
	const { get_schema, filter } = getContext('Meerkat');
	export let df: Writable;
	// TODO: Figure out if we should have a frontend_criteria
	// to control the frontend display of criteria, which allows
	// us to separate updating the backend store
	// from updating the frontend display.
	// This will allow us to selectively update the backend store
	// if certain criteria on the frontend are met (e.g. no-op if filters
	// are not valid.)
	// the downside is now we have to maintain two different filter criteria
	// and this can cause some lack of synchronization between the frontend and
	// backend.
	export let criteria: Writable<FilterCriterion[]>;
	export let operations: Writable<string[]>;
	export let title: Writable<string> = "";
	

	// Initialize the value to be the value of the store.
	// let criteria_frontend: FilterCriterion[] = $criteria;
	let criteria_frontend: FilterCriterion[] = [];
	criteria.subscribe((value) => {
		criteria_frontend = $criteria;
	});

	let schema_promise;
	let items_promise;
	$: {
		schema_promise = get_schema($df.ref_id);
		items_promise = schema_promise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => {
				return {
					value: column.name,
					label: column.name
				};
			});
		});
	}

	const trigger_filter = () => {
		// // We have to unpack the values of the dropdown values.
		// // TODO: Figure out how to bind the dropdown item.value only
		// let new_criteria = $criteria.map((criterion) => {
		// 	return {
		// 		// is_enabled is technically a frontend thing.
		// 		// but it simplifies the logic to keep the value
		// 		// in the backend store as well.
		// 		// This ensures the states are synchronized.
		// 		is_enabled: criterion.is_enabled,
		// 		column: criterion.column,
		// 		op: criterion.op,
		// 		value: criterion.value
		// 	};
		// });

		// console.log("New criteria", new_criteria)

		// Need to reset the array to trigger.
		console.log(criteria)
		criteria.set(criteria_frontend);
	};

	const onInputChange = (criterion: FilterCriterion, input_id: string, value: any) => {
		criterion[input_id] = value;
	};

	const setCheckbox = (
		criterion: FilterCriterion,
		value: boolean,
		ignore_check: boolean = false
	) => {
		// Setting to the same value, do nothing.
		criterion.is_enabled = value;
		trigger_filter();
	};

	const disableCheckbox = (criterion: FilterCriterion) => {
		return !criterion.column || !criterion.op || !criterion.value;
	};

	const addCriterion = () => {
		// Add a new filter criteria.
		criteria_frontend = [
			...criteria_frontend,
			{ is_enabled: false, column: '', op: $operations[0], value: '', is_fixed: false, source: 'frontend'}
		];
	};

	const deleteCriterion = (index: number) => {
		// Delete a filter criteria.
		// Store should only update if we are removing a criterion that is enabled.
		const is_enabled: boolean = criteria_frontend[index].is_enabled;
		criteria_frontend = criteria_frontend.filter((_, i) => i !== index);

		if (is_enabled) {
			criteria.set(criteria_frontend);
		}
	};

	const handleClear = () => {
		criteria.set([]);
	};
</script>

<div class="bg-slate-100 py-2 rounded-lg drop-shadow-md z-40 flex flex-col">
	<div class="flex space-x-6">
		{#if $title != ''}
			<div class="font-bold text-xl text-slate-600 self-start pl-2">
				{$title}
			</div>
		{/if}
		<div class="flex space-x-4 px-2">
			<button on:click={addCriterion} class="px-3 bg-violet-100 rounded-md text-violet-800 hover:drop-shadow-md">+ Add Filter</button>
			<button on:click={handleClear} class="px-3 bg-red-100 rounded-md text-red-800 hover:drop-shadow-md"> Clear </button>

		</div>
	</div>
	<div class="form-control w-full z-21">
		{#each criteria_frontend as criterion, i}
			<div class="py-2 input-group w-full flex items-center">
				<div class="px-1">
					<input
						id={'' + i}
						type="checkbox"
						disabled={disableCheckbox(criterion) || criterion.is_fixed}
						bind:checked={criterion.is_enabled}
						class="w-4 h-4 text-blue-600 bg-gray-100 rounded border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
						on:change={(e) => setCheckbox(criterion, e.target.checked, true)}
					/>
				</div>

				<div class="px-1 grow">
					{#await items_promise}
						<Select id="column" placeholder="...a column." loading={true} showChevron={true} />
					{:then items}
						<Select
							id="column"
							placeholder="...a column."
							value={criterion.column}
							{items}
							showChevron={true}
							on:change={(event) => onInputChange(criterion, 'column', event.detail.value)}
						/>
					{/await}
				</div>

				<div class="px-1">
					<Select
						id="op"
						placeholder="...an operation."
						value={criterion.op}
						items={$operations}
						showChevron={true}
						on:change={(event) => onInputChange(criterion, 'op', event.detail.value)}
					/>
				</div>

				<div class="px-1 grow-[1]">
					<input
						type="text"
						id="value"
						bind:value={criterion.value}
						placeholder="...a value"
						on:keypress={(e) => {
							if (e.charCode === 13 && !disableCheckbox(criterion)) {
								criterion.is_enabled = true;
								trigger_filter();
							}
						}}
						class="input-bordered w-full rounded-md shadow-md"
					/>
				</div>
				<div  class="px-1">
					<button class="" on:click={() => deleteCriterion(i)}>
						<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
							<path stroke-linecap="round" stroke-linejoin="round" d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>  
					</button>
				</div>
			</div>
		{/each}
	</div>
</div>

<style>
	/* 	
			CSS variables can be used to control theming.
			https://github.com/rob-balfre/svelte-select/blob/master/docs/theming_variables.md
	*/

	.themed {
		--itemPadding: 0.05rem;
		--itemColor: '#7c3aed';
		@apply rounded-md border-0;
		@apply z-[1000000];
	}
</style>

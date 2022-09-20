<script lang="ts">
	import type { Writable } from 'svelte/store';
	import type { FilterCriterion, DataPanelSchema } from '$lib/api/datapanel';
	import { getContext } from 'svelte';
	import Select from 'svelte-select';
	import { toast } from '@zerodevx/svelte-toast';

	const { get_schema, filter } = getContext('Interface');
	export let dp: Writable;
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
	export let operations: string[];

	// Initialize the value to be the value of the store.
	// let criteria_frontend: FilterCriterion[] = $criteria;
	$: criteria_frontend = $criteria;

	let schema_promise;
	let items_promise;
	$: {
		schema_promise = $get_schema($dp.box_id);
		items_promise = schema_promise.then((schema: DataPanelSchema) => {
			return schema.columns.map((column) => {
				return {
					value: column.name,
					label: column.name
				};
			});
		});
	}

	const trigger_filter = () => {
		console.log('triggering');
		console.log(criteria);
		console.log($criteria);

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
		criteria.set(criteria_frontend);
	};

	const onInputChange = (criterion: FilterCriterion, input_id: string, value: any) => {
		console.log(criterion, input_id, value);
		criterion[input_id] = value;
		console.log(criterion);
	};

	const setCheckbox = (
		criterion: FilterCriterion,
		value: boolean,
		ignore_check: boolean = false
	) => {
		// Setting to the same value, do nothing.
		console.log('setting checkbox', criterion.is_enabled, '->', value);
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
			{ is_enabled: false, column: '', op: operations[0], value: '' }
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
		// Toggle all filters to be off.
		let any_enabled: boolean = false;
		for (const criterion of criteria_frontend) {
			any_enabled = any_enabled || criterion.is_enabled;
			criterion.is_enabled = false;
		}
		if (any_enabled) {
			// Make a new array that svelte should respond to.
			criteria_frontend = [...criteria_frontend];
			criteria.set(criteria_frontend);
		}
	};

</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md">
	<div class="form-control w-full">
		{#each criteria_frontend as criterion, i}
			<div class="py-2 input-group w-100% flex items-center">
				<div class="px-3">
					<input
						id={'' + i}
						type="checkbox"
						disabled={disableCheckbox(criterion)}
						bind:checked={criterion.is_enabled}
						class="w-4 h-4 text-blue-600 bg-gray-100 rounded border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
						on:change={(e) => setCheckbox(criterion, e.target.checked, true)}
					/>
				</div>

				<div class="themed pr-2 w-48">
					{#await items_promise}
						<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
					{:then items}
						<Select
							id="column"
							placeholder="...a column."
							value={criterion.column}
							{items}
							showIndicator={true}
							listPlacement="auto"
							on:select={(event) => onInputChange(criterion, 'column', event.detail.value)}
						/>
					{/await}
				</div>

				<div class="themed pr-2">
					<Select
						id="op"
						placeholder="...an operation."
						value={criterion.op}
						items={operations}
						showIndicator={true}
						listPlacement="auto"
						on:select={(event) => onInputChange(criterion, 'op', event.detail.value)}
					/>
				</div>

				<div class="themed pr-5">
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
						class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
					/>
				</div>
				<div>
					<button class="themed" on:click={() => deleteCriterion(i)}>x</button>
				</div>
			</div>
		{/each}
		<div>
			<button on:click={addCriterion} class="px-3 hover:font-bold">+ Add Filter</button>
			<button on:click={handleClear} class="px-3 hover:font-bold">Clear All</button>
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

	.btn {
		@apply font-bold py-2 px-4 rounded;
	}
	.btn-blue {
		@apply bg-blue-500 text-white;
	}
	.btn-blue:hover {
		@apply bg-blue-700;
	}
</style>

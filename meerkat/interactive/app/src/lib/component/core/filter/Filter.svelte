<script lang="ts">
	import { fetchSchema } from '$lib/utils/api';
	import type { DataFrameRef, DataFrameSchema } from '$lib/utils/dataframe';
	import ClipboardPlus from 'svelte-bootstrap-icons/lib/ClipboardPlus.svelte';
	import Trash from 'svelte-bootstrap-icons/lib/Trash.svelte';
	import Select from 'svelte-select';
	import type { FilterCriterion } from './types';

	export let df: DataFrameRef;
	export let criteria: FilterCriterion[];
	export let operations: string[];
	export let title: string = '';

	let criteriaFrontend: FilterCriterion[] = [];
	$: criteriaFrontend = criteria;

	$: console.log('Criteria', criteria);

	let schemaPromise;
	let itemsPromise;
	$: {
		schemaPromise = fetchSchema({ df: df });
		itemsPromise = schemaPromise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => {
				return {
					value: column.name,
					label: column.name
				};
			});
		});
	}

	const triggerFilter = () => {
		// // We have to unpack the values of the dropdown values.
		// // TODO: Figure out how to bind the dropdown item.value only

		// Need to reset the array to trigger.
		console.log(criteria);
		criteria = criteriaFrontend;
	};

	const onInputChange = (criterion: FilterCriterion, input_id: string, value: any) => {
		criterion[input_id] = value;
	};

	const setCheckbox = (
		criterion: FilterCriterion,
		value: boolean,
		ignoreCheck: boolean = false
	) => {
		// Setting to the same value, do nothing.
		criterion.is_enabled = value;
		triggerFilter();
	};

	const disableCheckbox = (criterion: FilterCriterion) => {
		return !criterion.column || !criterion.op || !criterion.value;
	};

	const addCriterion = () => {
		// Add a new filter criteria.
		criteriaFrontend = [
			...criteriaFrontend,
			{
				is_enabled: false,
				column: '',
				op: operations[0],
				value: '',
				is_fixed: false,
				source: 'frontend'
			}
		];
	};

	const deleteCriterion = (index: number) => {
		// Delete a filter criteria.
		// Store should only update if we are removing a criterion that is enabled.
		const is_enabled: boolean = criteriaFrontend[index].is_enabled;
		criteriaFrontend = criteriaFrontend.filter((_, i) => i !== index);

		if (is_enabled) {
			criteria = criteriaFrontend;
		}
	};

	const handleClear = () => {
		criteria = [];
		criteriaFrontend = [];
	};
</script>

<div class="bg-slate-100 py-2 rounded-lg z-40 flex flex-col my-2">
	<div class="flex space-x-6">
		{#if title != ''}
			<div class="font-bold text-md text-slate-600 self-start pl-2">
				{title}
			</div>
		{/if}
		<div class="flex space-x-4 px-2">
			<button
				on:click={addCriterion}
				class="px-3 bg-slate-200 flex items-center gap-1.5 rounded-md text-slate-800 hover:drop-shadow-sm"
			>
				<ClipboardPlus /> Add Filter
			</button>
			<button
				on:click={handleClear}
				class="px-2 flex items-center gap-1.5 bg-slate-200 rounded-md text-slate-800 hover:drop-shadow-sm"
			>
				<Trash />
				Clear
			</button>
		</div>
	</div>
	<div class="form-control w-full z-21">
		{#each criteriaFrontend as criterion, i}
			<div class="py-2 input-group w-full flex items-center">
				<div class="px-1">
					<input
						id={'' + i}
						type="checkbox"
						disabled={disableCheckbox(criterion) || criterion.is_fixed}
						bind:checked={criterion.is_enabled}
						class="w-4 h-4 text-blue-600 bg-gray-100 rounded border-gray-300 focus:ring-blue-500 focus:ring-2"
						on:change={(e) => setCheckbox(criterion, e.target.checked, true)}
					/>
				</div>

				<div class="px-1 grow">
					{#await itemsPromise}
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
						items={operations}
						showChevron={true}
						on:change={(event) => onInputChange(criterion, 'op', event.detail.value)}
					/>
				</div>

				<div class="px-1 grow-[1]">
					<input
						type="text"
						id="value"
						value={criterion.value}
						placeholder="...a value"
						on:keypress={(e) => {
							criterion.value = e.target.value;
							if (e.charCode === 13 && !disableCheckbox(criterion)) {
								criterion.is_enabled = true;
								triggerFilter();
							}
						}}
						class="input-bordered w-full rounded-md shadow-md"
					/>
				</div>
				<div class="px-1">
					<button class="" on:click={() => deleteCriterion(i)}>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							stroke-width="1.5"
							stroke="currentColor"
							class="w-6 h-6"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
							/>
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

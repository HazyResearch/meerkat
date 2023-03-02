<script lang="ts">
	import { fetchSchema } from '$lib/utils/api';
	import type { DataFrameRef, DataFrameSchema } from '$lib/utils/dataframe';
	import { ClipboardPlus, Trash } from 'svelte-bootstrap-icons';
	import { dndzone } from 'svelte-dnd-action';
	import Select from 'svelte-select';
	import SvelteTooltip from 'svelte-tooltip';
	import { flip } from 'svelte/animate';
	import { v4 as uuidv4 } from 'uuid';
	import type { SortCriterion } from './types';

	export let df: DataFrameRef;
	export let criteria: SortCriterion[];
	export let title: string = '';

	// Users may change criteria rapidly on the frontend.
	// As such, sort needs to be performant.
	// We implement a smart debouncing mechanism that only triggers
	// the sort when criteria has meaningfully changed. It ignores cases where:
	//   1. The user changes an attribute of a criterion (e.g. value, ascending, etc.)
	//      to the same value.
	//   2. The user removes an inactive criterion.
	// In theory, all of this logic to skip the sort can (any maybe should) be
	// executed on the backend. However, this will require a roundtrip to the
	// backend every time an attribute of a criterion is changed, which can be slow.
	//
	// * One design is to have a frontend view of the criteria (i.e. criteria_frontend).
	// If the criteria changes from the backend, the view will be reset to the criteria.
	// This will ensure the frontend view is always up to date with the backend.
	// When the user interacts with the frontend, they will be manipulating the frontend view.
	// When the frontend view changes meaningfully, we will set the backend criteria to the
	// updated criteria.

	// Initialize the value to be the value of the store.
	// let criteria_frontend: FilterCriterion[] = $criteria;
	let criteriaFrontend: SortCriterion[] = [];

	// FIXME: Temporarily have to do this to update the frontend criteria
	// when the backend criteria changes.
	$: {
		criteriaFrontend = criteria;
	}
	// criteria.subscribe((value) => {
	// 	criteria_frontend = $criteria;
	// });

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

	const triggerSort = () => {
		// Need to reset the array to trigger.
		criteria = criteriaFrontend;
	};

	const onInputChange = (criterion: SortCriterion, input_id: string, value: any) => {
		const is_same_value = criterion[input_id] === value;
		criterion[input_id] = value;
		// Required for reactivity.
		criteriaFrontend = criteriaFrontend;
		if (!is_same_value) {
			criterion.is_enabled = true;
			triggerSort();
		}
	};

	const setCheckbox = (criterion: SortCriterion, value: boolean, ignoreCheck: boolean = false) => {
		// Setting to the same value, do nothing.fetchSchema
		criterion.is_enabled = value;
		triggerSort();
	};

	const setAscending = (criterion: SortCriterion, value: boolean) => {
		criterion.ascending = value;
		criterion.is_enabled = true;
		triggerSort();
	};

	const cellDisabled = (criterion: SortCriterion) => {
		return !criterion.column;
	};

	const addCriterion = () => {
		// Add a new filter criteria.
		const uuid_gen = uuidv4();
		criteriaFrontend = [
			{ id: uuid_gen, is_enabled: false, column: '', ascending: true },
			...criteriaFrontend
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
	};

	const flipDurationMs = 300;
	function handleDndConsider(e) {
		criteriaFrontend = e.detail.items;
	}
	function handleDndFinalize(e) {
		criteriaFrontend = e.detail.items;
		triggerSort();
	}
</script>

<div class="bg-slate-100 py-2 rounded-lg z-30 flex flex-col my-1">
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
				<ClipboardPlus /> Add Sort
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
	<div class="form-control w-full">
		<section
			use:dndzone={{ items: criteriaFrontend, flipDurationMs: flipDurationMs }}
			on:consider={handleDndConsider}
			on:finalize={handleDndFinalize}
		>
			{#each criteriaFrontend as criterion, i (criterion.id)}
				<div
					class="py-2 input-group w-full flex items-center"
					animate:flip={{ duration: flipDurationMs }}
				>
					<div class="px-1">
						<input
							id={'' + i}
							type="checkbox"
							disabled={cellDisabled(criterion)}
							bind:checked={criterion.is_enabled}
							class="w-4 h-4 text-blue-600 bg-gray-100 rounded border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
							on:change={(e) => setCheckbox(criterion, e.target.checked, true)}
						/>
					</div>

					<!-- Column selector -->
					<div class="themed px-1 grow">
						{#await itemsPromise}
							<Select id="column" placeholder="...a column." loading={true} showChevron={true} />
						{:then items}
							<Select
								id="column"
								placeholder="...a column."
								value={criterion.column}
								{items}
								showChevron={true}
								on:input={(event) => onInputChange(criterion, 'column', event.detail.value)}
							/>
						{/await}
					</div>

					<!-- Ascending / Descending button -->
					<div class="px-1">
						<button
							disabled={cellDisabled(criterion)}
							on:click={() => setAscending(criterion, !criterion.ascending)}
						>
							<SvelteTooltip
								tip={criterion.ascending ? 'Ascending' : 'Descending'}
								bottom
								color="#E2E8F0"
							>
								{#if criterion.ascending}
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
											d="M12 19.5v-15m0 0l-6.75 6.75M12 4.5l6.75 6.75"
										/>
									</svg>
								{:else}
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
											d="M19.5 13.5L12 21m0 0l-7.5-7.5M12 21V3"
										/>
									</svg>
								{/if}
							</SvelteTooltip>
						</button>
					</div>

					<!-- Delete button -->
					<div class="px-1">
						<button class="" on:click={() => deleteCriterion(i)}>
							<SvelteTooltip tip="Delete" bottom color="#E2E8F0">
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
							</SvelteTooltip>
						</button>
					</div>
				</div>
			{/each}
		</section>
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
</style>

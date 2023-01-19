<script lang="ts">
	import type { Writable } from 'svelte/store';
	import type { SortCriterion, DataFrameSchema } from '$lib/api/dataframe';
	import { getContext } from 'svelte';
	import Select from 'svelte-select';
	import { flip } from 'svelte/animate';
	import { dndzone } from 'svelte-dnd-action';
	import { v4 as uuidv4 } from 'uuid';
	import SvelteTooltip from 'svelte-tooltip';
	const { get_schema } = getContext('Meerkat');

	export let df: any;
	export let criteria: SortCriterion[];
	export let title: string = '';

	// Initialize the value to be the value of the store.
	// let criteria_frontend: FilterCriterion[] = $criteria;
	let criteria_frontend: SortCriterion[] = [];

	// FIXME: Temporarily have to do this to update the frontend criteria
	// when the backend criteria changes.
	$: {criteria_frontend = criteria}
	// criteria.subscribe((value) => {
	// 	criteria_frontend = $criteria;
	// });

	let schema_promise;
	let items_promise;
	$: {
		schema_promise = get_schema(df.ref_id);
		items_promise = schema_promise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => {
				return {
					value: column.name,
					label: column.name
				};
			});
		});
	}

	const trigger_sort = () => {
		// Need to reset the array to trigger.
		console.log("trigger sort", criteria_frontend);
		console.log("criteria", criteria);
		criteria = criteria_frontend;
	};

	const onInputChange = (criterion: SortCriterion, input_id: string, value: any) => {
		console.log("Input change", criterion);

		const is_same_value = criterion[input_id] === value;
		criterion[input_id] = value;
		// Required for reactivity.
		criteria_frontend = criteria_frontend;
		if (!is_same_value) {
			criterion.is_enabled = true;
			trigger_sort();
		}
	};

	const setCheckbox = (criterion: SortCriterion, value: boolean, ignore_check: boolean = false) => {
		// Setting to the same value, do nothing.
		criterion.is_enabled = value;
		trigger_sort();
	};

	const setAscending = (criterion: SortCriterion, value: boolean) => {
		criterion.ascending = value;
		criterion.is_enabled = true;
		trigger_sort();
	};

	const cellDisabled = (criterion: SortCriterion) => {
		return !criterion.column;
	};

	const addCriterion = () => {
		// Add a new filter criteria.
		const uuid_gen = uuidv4();
		criteria_frontend = [
			{ id: uuid_gen, is_enabled: false, column: '', ascending: true },
			...criteria_frontend
		];
	};

	const deleteCriterion = (index: number) => {
		// Delete a filter criteria.
		// Store should only update if we are removing a criterion that is enabled.
		const is_enabled: boolean = criteria_frontend[index].is_enabled;
		criteria_frontend = criteria_frontend.filter((_, i) => i !== index);

		if (is_enabled) {
			criteria = criteria_frontend;
		}
	};

	const handleClear = () => {
		criteria = [];
	};

	const flipDurationMs = 300;
	function handleDndConsider(e) {
		console.log('consider', e);
		criteria_frontend = e.detail.items;
	}
	function handleDndFinalize(e) {
		console.log('finalize', e);
		criteria_frontend = e.detail.items;
		trigger_sort();
	}
</script>

<div class="bg-slate-100 py-2 rounded-lg drop-shadow-md z-30 flex flex-col">
	<div class="flex space-x-6">
		{#if title != ''}
			<div class="font-bold text-xl text-slate-600 self-start pl-2">
				{title}
			</div>
		{/if}
		<div class="flex space-x-4 px-2">
			<button on:click={addCriterion} class="px-3 bg-violet-100 rounded-md text-violet-800 hover:drop-shadow-md">+ Add Sort</button>
			<button on:click={handleClear} class="px-3 bg-red-100 rounded-md text-red-800 hover:drop-shadow-md"> Clear </button>

		</div>
	</div>
	<div class="form-control w-full">
		<section
			use:dndzone={{ items: criteria_frontend, flipDurationMs: flipDurationMs }}
			on:consider={handleDndConsider}
			on:finalize={handleDndFinalize}
		>
			{#each criteria_frontend as criterion, i (criterion.id)}
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
						{#await items_promise}
							<Select
								id="column"
								placeholder="...a column."
								loading={true}
								showChevron={true}
							/>
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
								tip={criterion.ascending ? 'ascending' : 'descending'}
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

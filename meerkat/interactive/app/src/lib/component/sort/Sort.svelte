<script lang="ts">
	import type { Writable } from 'svelte/store';
	import type { SortCriterion, DataPanelSchema } from '$lib/api/datapanel';
	import { getContext } from 'svelte';
	import Select from 'svelte-select';
	import {flip} from "svelte/animate";
    import {dndzone} from "svelte-dnd-action";
    import { v4 as uuidv4 } from 'uuid';

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
	export let criteria: Writable<SortCriterion[]>;
	export let operations: string[];

	// Initialize the value to be the value of the store.
	// let criteria_frontend: FilterCriterion[] = $criteria;
	let criteria_frontend: SortCriterion[] = [];
	criteria.subscribe((value) => {
		criteria_frontend = $criteria;
	});

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

	const trigger_sort = () => {
		// Need to reset the array to trigger.
		criteria.set(criteria_frontend);
	};

	const onInputChange = (criterion: SortCriterion, input_id: string, value: any) => {
        const is_same_value = criterion[input_id] === value;
		criterion[input_id] = value;
        // Required for reactivity.
        criteria_frontend = criteria_frontend;
        if (!is_same_value) {
            criterion.is_enabled = true;
            trigger_sort();
        }
	};

	const setCheckbox = (
		criterion: SortCriterion,
		value: boolean,
		ignore_check: boolean = false
	) => {
		// Setting to the same value, do nothing.
		console.log('setting checkbox', criterion.is_enabled, '->', value);
		criterion.is_enabled = value;
		trigger_sort();
	};

    const setAscending = (
        criterion: SortCriterion,
        value: boolean
    ) => {
        criterion.ascending = value;
        criterion.is_enabled = true;
        trigger_sort();
    }

	const cellDisabled = (criterion: SortCriterion) => {
        console.log("not criterion col:", !criterion.column);
		return !criterion.column;
	};

	const addCriterion = () => {
		// Add a new filter criteria.
        const uuid_gen = uuidv4(); 
		criteria_frontend = [
            { id: uuid_gen, is_enabled: false, column: '', ascending: true, },
			...criteria_frontend
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

    const flipDurationMs = 300;
    function handleDndConsider(e) {
        console.log("consider", e);
        criteria_frontend = e.detail.items;
    }
    function handleDndFinalize(e) {
        console.log("finalize", e);
        criteria_frontend = e.detail.items;
        trigger_sort();
    }

</script>

<div class="bg-slate-100 py-2 rounded-lg drop-shadow-md">
	<div class="form-control w-full">
        <section use:dndzone={{items: criteria_frontend, flipDurationMs: flipDurationMs}} on:consider={handleDndConsider} on:finalize={handleDndFinalize}>
		{#each criteria_frontend as criterion, i (criterion.id)}
			<div class="py-2 input-group w-full flex items-center" animate:flip="{{duration: flipDurationMs}}">
				<div class="px-3">
					<input
						id={'' + i}
						type="checkbox"
                        disabled='{cellDisabled(criterion)}'
						bind:checked={criterion.is_enabled}
						class="w-4 h-4 text-blue-600 bg-gray-100 rounded border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
						on:change={(e) => setCheckbox(criterion, e.target.checked, true)}
					/>
				</div>

				<div class="themed pr-2 w-flex">
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
					<button disabled='{cellDisabled(criterion)}' on:click={() => setAscending(criterion, !criterion.ascending)}>
                        {#if criterion.ascending}
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M12 19.5v-15m0 0l-6.75 6.75M12 4.5l6.75 6.75" />
                            </svg>
                        {:else}
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 13.5L12 21m0 0l-7.5-7.5M12 21V3" />
                            </svg>
                          
                        {/if}
                    </button>
				</div>
				<div>
					<button class="themed" on:click={() => deleteCriterion(i)}>x</button>
				</div>
			</div>
		{/each}
        </section>
		<div>
			<button on:click={addCriterion} class="px-3 hover:font-bold">+ Add Sort</button>
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

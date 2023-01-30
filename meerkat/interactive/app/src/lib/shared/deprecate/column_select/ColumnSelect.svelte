<script lang="ts">
	import { type DataFrameSchema } from '$lib/api/dataframe';
	import { createEventDispatcher } from 'svelte';
	import Select from 'svelte-select';

	const dispatch = createEventDispatcher();

	export let schema_promise: Promise<DataFrameSchema>;
	// export let refresh_callback: RefreshCallback;
	// export let match_criterion: MatchCriterion;

	export let column: string = '';

	// let on_search = async () => {
	// 	if (column === '') {
	// 		status = 'error';
	// 		return;
	// 	}
	// 	status = 'working';
	// 	match_criterion = new MatchCriterion(column, search_box_text);

	// 	let promise = refresh_callback();
	// 	promise.then(() => {
	// 		status = 'success';
	// 		// Dispatch the match event
	// 		dispatch_match(search_box_text, column);
	// 	});
	// };

	let items_promise = schema_promise.then((schema) => {
		return schema.columns.map((column) => {
			return {
				value: column.name,
				label: column.name
			};
		});
	});

	function handleSelect(event) {
		column = event.detail.value;
		dispatch('select');
	}

	function handleClear() {
		column = '';
	}
</script>

<div class="py-3 rounded-lg drop-shadow-md">
	<div class="themed pr-2 pl-3">
		{#await items_promise}
			<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
		{:then items}
			<Select
				id="column"
				placeholder="...a column."
				{items}
				showIndicator={true}
				listPlacement="auto"
				on:select={handleSelect}
				on:clear={handleClear}
			/>
		{/await}
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
		@apply rounded-md w-80 border-0;
	}
</style>

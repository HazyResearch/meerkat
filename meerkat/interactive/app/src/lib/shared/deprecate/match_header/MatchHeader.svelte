<script lang="ts">
	import type { RefreshCallback } from '$lib/shared/deprecate/callbacks';
	import type { DataFrameSchema } from '$lib/api/dataframe';
	import Status from '$lib/shared/common/Status.svelte';
	import { createEventDispatcher } from 'svelte';
	import Select from 'svelte-select';

	export class MatchCriterion {
		constructor(readonly column: string, readonly query: string) { }
	}


	const dispatch = createEventDispatcher();

	function dispatch_match(search_text: string, column: string) {
		dispatch('match', {
			search_text: search_text,
			column: column
		});
	}

	export let schema_promise: Promise<DataFrameSchema>;
	export let refresh_callback: RefreshCallback;
	export let match_criterion: MatchCriterion;

	export let search_box_text: string = '';
	export let column: string = '';
	export let status: string = 'waiting';

	let on_search = async () => {
		if (column === '') {
			status = 'error';
			return;
		}
		status = 'working';
		match_criterion = new MatchCriterion(column, search_box_text);

		let promise = refresh_callback();
		promise.then(() => {
			status = 'success';
			// Dispatch the match event
			dispatch_match(search_box_text, column);
		});
	};

	const onKeyPress = (e) => {
		if (e.charCode === 13) on_search();
		else status = 'waiting';
	};

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
	}

	function handleClear() {
		column = '';
	}
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md">
	<div class="form-control w-full">
		<div class="input-group w-full flex items-center">
			<div class="px-3">
				<Status {status} />
			</div>
			<input
				type="text"
				bind:value={search_box_text}
				placeholder="Write some text to be matched..."
				class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
				on:keypress={onKeyPress}
			/>
			<div class="text-slate-400 px-2">against</div>

			<div class="themed pr-2">
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

<script lang="ts">
	import { match, MatchCriterion, sort, type DataPanelSchema } from '$lib/api/datapanel';
	import type { RefreshCallback } from '$lib/api/callbacks';
	import { api_url } from '$lib/../routes/network/stores';
	import Select from 'svelte-select';
	import Status from '$lib/components/common/Status.svelte';


	export let schema_promise: Promise<DataPanelSchema>;
	export let refresh_callback: RefreshCallback;
	export let match_criterion: MatchCriterion;

	let search_box_text: string = '';
	let search_promise: Promise<DataPanelSchema>;
	let column: string = '';
	let status: string = 'waiting';

	let on_search = async () => {
		if (column === '') {
			console.log('empty');
			status = "error";
			return;
		}
		match_criterion = new MatchCriterion(column, search_box_text);

		let promise = refresh_callback();
		
		promise.then(() => {
			status = 'success';
		})
		status = 'success';
	};

	const onKeyPress = (e) => {
		if (e.charCode === 13) on_search();
		else status = 'waiting';
	};

	const empty_items: Array<any> = [];
	let items_promise = schema_promise.then((schema) => {
		return schema.columns.map((column) => {
			return {
				value: column.name,
				label: column.name
			};
		});
	});

	let favouriteFood = undefined;

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
				{#await search_promise}
					<Status status="working" />
				{:then items}					
					<Status status={status} />
				{/await}
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

<script lang="ts">
	import Status from '$lib/shared/common/Status.svelte';
	import { dispatch } from '$lib/utils/api';
	import type { DataFrameRef, DataFrameSchema } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import Select from 'svelte-select';
	import Textbox from '../textbox/Textbox.svelte';

	export let df: DataFrameRef;
	export let against: string;
	export let text: string;
	export let title: string = '';
	export let showAgainst: boolean = false;

	export let onMatch: Endpoint;
	export let getMatchSchema: Endpoint;

	let status: string = 'waiting';

	let schemaPromise;
	let itemsPromise;
	$: {
		schemaPromise = dispatch(getMatchSchema.endpointId, { detail: {} });
		itemsPromise = schemaPromise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => ({ value: column.name, label: column.name }));
		});
	}

	const onKeyPress = (e: CustomEvent) => {
		console.log('onkeypress', e);
		if (e.detail.code === 'Enter') {
			console.log('matching');
			onSearch();
		} else status = 'waiting';
	};

	let onSearch = async () => {
		if (against === '') {
			status = 'error';
			return;
		}
		status = 'working';
		let promise = dispatch(onMatch.endpointId, {
			detail: { against: against, query: text }
		});
		promise
			.then(() => {
				status = 'success';
			})
			.catch((error: TypeError) => {
				status = 'error';
			});
	};

	function handleSelect(event) {
		against = event.detail.value;
	}

	function handleClear() {
		against = '';
	}
	$: againstItem = { value: against, label: against };
</script>

<div class="bg-slate-100 py-1 rounded-lg z-50 flex flex-col my-1">
	{#if title != ''}
		<div class="font-bold text-md text-slate-600 pl-2 text-center">
			{title}
		</div>
	{/if}
	<div class="form-control">
		<div class="input-group w-100% flex items-center px-3 space-x-2">
			<div class="">
				<Status {status} />
			</div>
			<Textbox bind:text on:keyup={onKeyPress} />

			{#if showAgainst}
				<div class="text-slate-400 px-2">against</div>

				<div class="themed pr-2 w-48">
					{#await itemsPromise}
						<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
					{:then items}
						<Select
							id="column"
							placeholder="...a column."
							value={againstItem}
							{items}
							showIndicator={true}
							listPlacement="auto"
							on:select={handleSelect}
							on:clear={handleClear}
						/>
					{/await}
				</div>
			{/if}
		</div>
	</div>
</div>

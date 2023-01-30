<script lang="ts">
	import type { DataFrameSchema } from '$lib/api/dataframe';
	import { getContext } from 'svelte';
	import Status from '$lib/shared/common/Status.svelte';
	import Select from 'svelte-select';
	import type { Endpoint } from '$lib/utils/types';

	const { dispatch } = getContext('Meerkat');

	export let df: any;
	export let against: string;
	export let on_match: Endpoint;
	export let text: string;
	export let title: string = '';
	export let get_match_schema: Endpoint;

	let status: string = 'waiting';

	let schema_promise;
	let items_promise;
	$: {
		schema_promise = dispatch(get_match_schema.endpoint_id, { detail: {} });
		items_promise = schema_promise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => ({ value: column.name, label: column.name }));
		});
	}

	const onKeyPress = (e) => {
		if (e.charCode === 13) on_search();
		else status = 'waiting';
	};

	let on_search = async () => {
		if (against === '') {
			status = 'error';
			return;
		}
		status = 'working';
		let promise = dispatch(on_match.endpoint_id, {
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
	$: against_item = { value: against, label: against };
</script>

<div class="bg-slate-100 py-1 rounded-lg drop-shadow-md z-50 flex flex-col">
	{#if title != ''}
		<div class="font-bold text-md text-slate-600 pl-2 text-center">
			{title}
		</div>
	{/if}
	<div class="form-control">
		<div class="input-group w-100% flex items-center">
			<div class="px-3">
				<Status {status} />
			</div>
			<input
				type="text"
				bind:value={text}
				placeholder="Write some text to be matched..."
				class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
				on:keypress={onKeyPress}
			/>
			<div class="text-slate-400 px-2">against</div>

			<div class="themed pr-2 w-48">
				{#await items_promise}
					<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
				{:then items}
					<Select
						id="column"
						placeholder="...a column."
						value={against_item}
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
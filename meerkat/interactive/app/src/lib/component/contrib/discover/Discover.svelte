<script lang="ts">
	import type { DataFrameRef, DataFrameSchema } from '$lib/api/dataframe';
	import Status from '$lib/shared/common/Status.svelte';
	import type { Endpoint } from '$lib/utils/types';
	import { getContext } from 'svelte';
	import Select from 'svelte-select';

	const { dispatch } = getContext('Meerkat');

	export let df: DataFrameRef;
	export let by: string;
	export let on_discover: Endpoint;
	export let get_discover_schema: Endpoint;

	let status: string = 'waiting';

	let schema_promise;
	let items_promise;
	$: {
		schema_promise = dispatch(get_discover_schema.endpoint_id, { detail: {} });
		items_promise = schema_promise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => ({ value: column.name, label: column.name }));
		});
	}

	const on_submit = async () => {
		if (by === '') {
			status = 'error';
			return;
		}
		status = 'working';
		let promise = dispatch(on_discover.endpoint_id, {
			detail: {
				by: by
			}
		});
		promise
			.then(() => {
				status = 'success';
			})
			.catch((error: TypeError) => {
				status = 'error';
				console.log(error);
			});
	};

	function handleSelect(event) {
		by = event.detail.value;
	}

	function handleClear() {
		by = '';
	}
	$: by_item = { value: by, label: by };
</script>

<div class="bg-slate-100 py-1 rounded-lg drop-shadow-md z-50 flex flex-col">
	<div class="form-control">
		<div class="input-group w-100% flex items-center space-x-3">
			<div class="px-3">
				<Status {status} />
			</div>
			<button
				on:click={on_submit}
				class="px-3 bg-violet-100 rounded-md text-violet-800 hover:drop-shadow-md"
			>
				Discover
			</button>
			<div class="themed pr-2 w-48">
				{#await items_promise}
					<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
				{:then items}
					<Select
						id="column"
						placeholder="...a column."
						value={by_item}
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

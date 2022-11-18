<script lang="ts">
	import type { Writable } from 'svelte/store';
	import type { DataFrameSchema } from '$lib/api/dataframe';
	import { getContext } from 'svelte';
	import Status from '$lib/shared/common/Status.svelte';
	import Select from 'svelte-select';
	import type { Endpoint } from '$lib/utils/types';

	const { dispatch } = getContext('Interface');

	export let df: Writable;
	export let against: Writable<string>;
	export let on_discover: Endpoint;
	export let get_discover_schema: Endpoint;

	let status: string = 'waiting';

	let schema_promise;
	let items_promise;
	$: {
		schema_promise = $dispatch(get_discover_schema.endpoint_id, {}, {});
		items_promise = schema_promise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => ({ value: column.name, label: column.name }));
		});
	}

	const on_submit = async () => {
		if ($against === '') {
			status = 'error';
			return;
		}
		status = 'working';
		let promise = $dispatch(
			on_discover.endpoint_id,
			{
				against: $against
			},
			{}
		);
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
		$against = event.detail.value;
	}

	function handleClear() {
		$against = '';
	}
	$: against_item = { value: $against, label: $against };
</script>

<div class="bg-slate-100 py-1 rounded-lg drop-shadow-md z-50 flex flex-col">
	<div class="form-control">
		<div class="input-group w-100% flex items-center">
			<div class="px-3">
				<Status {status} />
			</div>
			<button
				on:click={on_submit}
				class="px-3 bg-violet-100 rounded-md text-violet-800 hover:drop-shadow-md"
			>
				discover
			</button>
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

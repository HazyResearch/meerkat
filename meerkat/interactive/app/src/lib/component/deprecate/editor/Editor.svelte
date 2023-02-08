<script lang="ts">
	import Status from '$lib/shared/common/Status.svelte';
	import { fetchSchema } from '$lib/utils/api';
	import type { DataFrameRef, DataFrameSchema } from '$lib/utils/dataframe';
	import type { EditTarget } from '$lib/utils/types';
	import Select from 'svelte-select';
	import { get, type Writable } from 'svelte/store';


	export let df: DataFrameRef;
	export let target: EditTarget;
	export let primary_key: string;
	export let selected: Writable<Array<number>>;
	export let col: Writable<string>;
	export let text: Writable<string>;
	export let title: string = '';

	let status: string = 'waiting';

	let schemaPromise: any;
	let itemsPromise: any;

	$: target.target = get(target.target);

	$: {
		schemaPromise = fetchSchema(target.target);
		itemsPromise = schemaPromise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => {
				return {
					value: column.name,
					label: column.name
				};
			});
		});
	}

	const onKeyPress = (e) => {
		if (e.charCode === 13) onEdit();
		else status = 'waiting';
	};

	let onEdit = async () => {
		if ($col === '') {
			status = 'error';
			return;
		}
		status = 'working';

		let modifications_promise;
		if (primary_key === undefined) {
			modifications_promise = edit_target($df.ref_id, target, $text, $col, $selected);
		} else {
			modifications_promise = edit_target(
				$df.ref_id,
				target,
				$text,
				$col,
				null,
				$selected,
				primary_key
			);
		}

		modifications_promise
			.then(() => {
				status = 'success';
			})
			.catch((error: TypeError) => {
				status = 'error';
				console.log(error);
			});
	};

	function handle_select(event) {
		$col = event.detail.value;
	}

	function handle_clear() {
		$col = '';
	}
	$: col_item = { value: $col, label: $col };

	let select_div;
</script>

<div class="w-full items-center bg-slate-100 py-1 rounded-lg drop-shadow-md z-20">
	{#if title != ''}
		<div class="font-bold text-md text-slate-600 self-start pl-2 text-center">
			{title}
		</div>
	{/if}
	<div class="flex items-center">
		<div class="px-3">
			<Status {status} />
		</div>
		<input
			type="text"
			bind:value={$text}
			placeholder="Enter a value..."
			class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
			on:keypress={onKeyPress}
		/>
		<div class="text-slate-400 px-2">for</div>

		<div class="themed pr-2 w-48" bind:this={select_div}>
			{#await itemsPromise}
				<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
			{:then items}
				<Select
					id="column"
					placeholder="...a column."
					value={col_item}
					{items}
					showIndicator={true}
					listPlacement="auto"
					on:select={handle_select}
					on:clear={handle_clear}
					appendListTarget={select_div}
				/>
			{/await}
		</div>
		
		
	</div>
</div>

<script lang="ts">
	import { getContext } from 'svelte';
	import Status from '$lib/shared/common/Status.svelte';
	import Select from 'svelte-select';
	import type { EditTarget } from '$lib/utils/types';
	import { get, type Writable } from 'svelte/store';

	const { get_schema, edit_target } = getContext('Interface');

	export let df: Writable;
	export let target: EditTarget;
	export let primary_key: string;
	export let selected: Writable<Array<number>>;
	export let col: Writable<string>;
	export let text: Writable<string>;

	let status: string = 'waiting';

	let schema_promise: any;
	let items_promise: any;

	$: target.target = get(target.target);

	$: {
		schema_promise = $get_schema(target.target.box_id);
		items_promise = schema_promise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => {
				return {
					value: column.name,
					label: column.name
				};
			});
		});
	}

	const on_key_press = (e) => {
		if (e.charCode === 13) on_edit();
		else status = 'waiting';
	};

	let on_edit = async () => {
		if ($col === '') {
			status = 'error';
			return;
		}
		status = 'working';

		let modifications_promise;
		if (primary_key === undefined) {
			modifications_promise = $edit_target($df.box_id, target, $text, $col, $selected);
		} else {
			modifications_promise = $edit_target(
				$df.box_id,
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

<div class="w-full flex items-center bg-slate-100 py-3 rounded-lg drop-shadow-md z-20">
	<div class="px-3">
		<Status {status} />
	</div>
	<div class="themed pr-2 w-48" bind:this={select_div}>
		{#await items_promise}
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
	<input
		type="text"
		bind:value={$text}
		placeholder="Enter a value..."
		class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
		on:keypress={on_key_press}
	/>
</div>

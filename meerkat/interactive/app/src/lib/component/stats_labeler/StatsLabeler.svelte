<script lang="ts">
	import { getContext } from 'svelte';
	import Status from '$lib/components/common/Status.svelte';
	import Select from 'svelte-select';
	import type { EditTarget } from '$lib/utils/types';
	import { get, type Writable } from 'svelte/store';

	const { get_schema, edit_target } = getContext('Interface');

	export let dp: Writable;
	export let target: EditTarget;
	export let primary_key: string;
	export let selected: Writable<Array<number>>;
	export let col: Writable<string>;
	export let text: Writable<string>;
	export let mode: Writable<string>;

	let status: string = 'waiting';

	let schema_promise: any;
	let items_promise: any;

	$: target.target = get(target.target);

	$: {
		schema_promise = $get_schema(target.target.box_id);
		items_promise = schema_promise.then((schema: DataPanelSchema) => {
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

		// All labeled examples should be part of the test set.
		let metadata = null;
		if ($mode === 'train') {
			metadata = {"split": {"value": "train", "default": ""}};
		}
		console.log(metadata)
		let modifications_promise;
		if (primary_key === undefined) {
			modifications_promise = $edit_target(
				$dp.box_id,
				target,
				$text,
				$col,
				$selected,
				null,
				null,
				metadata
			);
		} else {
			modifications_promise = $edit_target(
				$dp.box_id,
				target,
				$text,
				$col,
				null,
				$selected,
				primary_key,
				metadata
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

	// Modes
	function handle_mode_select(event) {
		$mode = event.detail.value;
	}
	$: mode_item = { value: $mode, label: $mode };
	const modes = ['train', 'precision', 'recall']

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
		type="number"
		bind:value={$text}
		placeholder="Enter a value..."
		class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
		on:keypress={on_key_press}
	/>
	<div class="themed pr-2 w-48">
		<Select
			id="mode"
			value={mode_item}
			placeholder="...a mode."
			items={modes}
			isClearable={false}
			showIndicator={true}
			listPlacement="auto"
			on:select={handle_mode_select}
		/>
	</div>
</div>

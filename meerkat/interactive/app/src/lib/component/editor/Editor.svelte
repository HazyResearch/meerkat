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

	const onKeyPress = (e) => {
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
			modifications_promise = $edit_target($dp.box_id, target, $text, $col, $selected);
		} else {
			modifications_promise = $edit_target(
				$dp.box_id,
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

	function handleSelect(event) {
		$col = event.detail.value;
	}

	function handleClear() {
		$col = '';
	}
	$: col_item = { value: $col, label: $col };
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md">
	<div class="form-control w-full">
		<div class="input-group w-100% flex items-center">
			<div class="px-3">
				<Status {status} />
			</div>
			<div class="themed pr-2 w-48">
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
						on:select={handleSelect}
						on:clear={handleClear}
					/>
				{/await}
			</div>
			<input
				type="number"
				bind:value={$text}
				placeholder="Enter a value..."
				class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
				on:keypress={onKeyPress}
			/>
		</div>
	</div>
</div>

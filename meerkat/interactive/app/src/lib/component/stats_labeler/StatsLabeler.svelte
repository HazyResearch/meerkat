<script lang="ts">
	import { getContext } from 'svelte';
	import EraserFill from 'svelte-bootstrap-icons/lib/EraserFill.svelte';
	import X from 'svelte-bootstrap-icons/lib/X.svelte';
	import Check2 from 'svelte-bootstrap-icons/lib/Check2.svelte';

	import Status from '$lib/components/common/Status.svelte';
	import type { EditTarget } from '$lib/utils/types';
	import { get, type Writable } from 'svelte/store';
	import Phases from './Phases.svelte';

	const { edit_target, get_rows } = getContext('Interface');

	export let dp: Writable;
	export let label_target: EditTarget;
	export let phase_target: EditTarget;
	export let primary_key: string;
	export let phase: Writable<string>;
	export let selected: Writable<Array<number>>;
	export let active_key: Writable<string>;

	$: col = `label(${$active_key})`;

	let status: string = 'waiting';

	$: label_target.target = get(label_target.target);
	$: phase_target.target = get(phase_target.target);


	let counts_promise: any;
	$: {
		$dp; // needed to trigger on dp change
		counts_promise = $get_rows(
			label_target.target.box_id,
			undefined,
			undefined,
			undefined,
			[primary_key, col],
			primary_key,
			$selected
		).then((rows: any) => {
			return count_values(rows.rows);
		});
	}

	let on_edit = async (value: number) => {
		if (col === '') {
			status = 'error';
			return;
		}
		if ($selected.length == 0) {
			return;
		}
		status = 'working';

		// All labeled examples should be part of the test set.
		let metadata = null;
		if ($phase === 'train') {
			metadata = { split: { value: 'train', default: '' } };
		} else {
			metadata = { split: { value: 'test', default: '' } };
		}

		let modifications_promise;
		if (primary_key === undefined) {
			modifications_promise = $edit_target(
				$dp.box_id,
				label_target,
				value,
				col,
				$selected,
				null,
				null,
				metadata
			);
		} else {
			modifications_promise = $edit_target(
				$dp.box_id,
				label_target,
				value,
				col,
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

	// Modes

	function handle_phase_change(event: CustomEvent) {
		let new_phase: string = event.detail;
		$edit_target(
			phase_target.target.box_id,
			phase_target,
			new_phase,
			col="phase",
			null,
			[$active_key],
			"key",
			null 
		)
	}

	function count_values(values: Array<number>) {
		let counts = { '-1': 0, '0': 0, '1': 0 };
		for (let i = 0; i < values.length; i++) {
			let num = values[i][1];
			counts[num] = counts[num] ? counts[num] + 1 : 1;
		}
		return counts;
	}
</script>

<div class="w-full bg-slate-100 py-2 rounded-lg drop-shadow-md z-20">
	<div class="font-bold text-xl text-slate-600 text-center w-full">
		<!-- TODO(Sabri): This should be a customizable name in the future. -->
		Attribution
	</div>

	<div class="flex items-center">
		{#if $active_key === "_no_selection"}
			<div class="text-center w-full text-slate-400">No slice selected.</div>
		{:else}
			<Phases on:phase_change={handle_phase_change} active_phase={$phase}/>
			<div class="px-3">
				<Status {status} />
			</div>
			<!-- <div class="themed pr-2 w-26">
				<Select
					id="mode"
					value={mode_item}
					placeholder="...a mode."
					items={modes}
					clearable={false}
					listAutoWidth={false}
					listPlacement="auto"
					on:select={handle_mode_select}
				/>
			</div> -->

			<div class="flex space-x-1">
				<button
					on:click={() => on_edit(-1)}
					class="flex items-center justify-center group w-10 h-8 rounded-lg bg-violet-200 text-violet-800 hover:shadow-lg "
				>
					<EraserFill class="" width={24} height={24} />
				</button>
				<button
					on:click={() => on_edit(0)}
					class="flex items-center justify-center group w-10 h-8 rounded-lg bg-violet-200 text-violet-800 hover:shadow-lg "
				>
					<X class="" width={24} height={24} />
				</button>
				<button
					on:click={() => on_edit(1)}
					class="flex items-center justify-center group w-10 h-8 rounded-lg bg-violet-200 text-violet-800 hover:shadow-lg "
				>
					<Check2 class="" width={24} height={24} />
				</button>
			</div>
			{#await counts_promise}
				Loading...
			{:then counts}
				<div class="flex space-x-2">
					<div>{counts['-1']}</div>
					<div>{counts['0']}</div>
					<div>{counts['1']}</div>
				</div>
			{/await}
			<!-- <input
				type="number"
				bind:value={$text}
				placeholder="Enter a value..."
				class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
				on:keypress={on_key_press}
			/> -->
		{/if}
	</div>
</div>

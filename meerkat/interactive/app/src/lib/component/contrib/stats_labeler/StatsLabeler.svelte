<script lang="ts">
	import Check2 from 'svelte-bootstrap-icons/lib/Check2.svelte';
	import EraserFill from 'svelte-bootstrap-icons/lib/EraserFill.svelte';
	import X from 'svelte-bootstrap-icons/lib/X.svelte';

	import Interval from '$lib/shared/cell/interval/Interval.svelte';
	import { fetchChunk } from '$lib/utils/api';
	import type { EditTarget } from '$lib/utils/types';
	import { get, type Writable } from 'svelte/store';
	import Phases from './Phases.svelte';

	export let df: Writable;
	export let label_target: EditTarget;
	export let phase_target: EditTarget;
	export let primary_key: string;
	export let phase: Writable<string>;
	export let selected: Writable<Array<number>>;
	export let active_key: Writable<string>;
	export let precision_estimate: Writable<Array<number>>;
	export let recall_estimate: Writable<Array<number>>;

	$: col = `label(${$active_key})`;

	let status: string = 'waiting';

	$: label_target.target = get(label_target.target);
	$: phase_target.target = get(phase_target.target);

	$: console.log(precision_estimate)

	let counts_promise: any;
	$: {
		$df; // needed to trigger on df change
		console.log(col)
		counts_promise = fetchChunk(
			label_target.target.ref_id,
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
			modifications_promise = edit_target(
				$df.ref_id,
				label_target,
				value,
				col,
				$selected,
				null,
				null,
				metadata
			);
		} else {
			modifications_promise = edit_target(
				$df.ref_id,
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
		edit_target(
			phase_target.target.ref_id,
			phase_target,
			new_phase,
			'phase',
			null,
			[$active_key],
			'key',
			null
		);
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

<div class="flex flex-col space-y-3 w-full bg-slate-100 py-2 rounded-lg drop-shadow-md z-20">
	<div class="flex px-5 space-x-3">
		<div class="font-bold text-xl text-slate-600 ">
			<!-- TODO(Sabri): This should be a customizable name in the future. -->
			Attribution
		</div>
		<Phases on:phase_change={handle_phase_change} active_phase={$phase} />
	</div>

	<div class="flex items-center space-x-5 px-10">
		{#if $active_key === '_no_selection'}
			<div class="text-center w-full text-slate-400">No slice selected.</div>
		{:else}
			<!-- <div class="px-3">
				<Status {status} />
			</div> -->

			
			<div class="grid grid-cols-[auto_auto_auto] grid-rows-3 grid-flow-col gap-y-1 gap-x-2">
				<div class="text-slate-500 font-bold">Positive</div>
				<div class="text-slate-500 font-bold">Negative</div>
				<div class="text-slate-500 font-bold">Unlabeled</div>
				{#each ["1", "0", "-1"] as num}
					<div class="flex font-bold text-slate-500 w-6">
						{#await counts_promise}
							<div class="text-center">-</div>
						{:then counts}
							<div class="text-center">{counts[num]}</div>
						{/await}
					</div>

				{/each}
				<button
					on:click={() => on_edit(1)}
					class="flex items-center justify-center group w-10 h-6 rounded-lg bg-violet-200 text-violet-800 hover:shadow-lg "
				>
					<Check2 class="" width={24} height={24} />
				</button>
				<button
					on:click={() => on_edit(0)}
					class="flex items-center justify-center group w-10 h-6 rounded-lg bg-violet-200 text-violet-800 hover:shadow-lg "
				>
					<X class="" width={24} height={24} />
				</button>
				<button
					on:click={() => on_edit(-1)}
					class="flex items-center justify-center group w-10 h-6 rounded-lg bg-violet-200 text-violet-800 hover:shadow-lg "
				>
					<EraserFill class="" width={24} height={24} />
				</button>
				
			</div>

			<div class="bg-white rounded-md flex flex-col shadow-lg  h-full">
				<div class="text-slate-400 px-3 py-1 self-center">Precision</div>
				{#if $precision_estimate === null}
					<div class="text-center text-slate-400 px-3 py-1 self-center">No estimate.</div>
				{:else}
					<div class="font-bold text-2xl px-3 self-center">
						<Interval data={
							$precision_estimate
						} percentage={true} />
					</div>
				{/if}
			</div>
			<div class="bg-white rounded-md flex flex-col shadow-lg h-full">
				<div class="text-slate-400 px-3 py-1 self-center">Recall</div>
				{#if $recall_estimate === null}
					<div class="text-center text-slate-400 px-3 py-1 self-center">No estimate.</div>
				{:else}
					<div class="font-bold text-2xl px-3 self-center">
						<Interval data={
							$recall_estimate
						} percentage={true} />
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>

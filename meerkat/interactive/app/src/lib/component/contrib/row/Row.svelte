<script lang="ts">
	import Cell from '$lib/shared/cell/Cell.svelte';
	import { dispatch, fetchChunk, fetchSchema } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import { Scissors } from 'svelte-bootstrap-icons';

	export let df: DataFrameRef;
	export let selected_key: string;
	export let columns: string[] = [];
	export let stats: Record<string, any> = {};
	export let rename: Record<string, string> = {};
	export let title: string = '';

	export let on_change: Endpoint | null = null;
	export let on_slice_creation: Endpoint | null = null;

	$: schema_promise = fetchSchema({ df: df });

	$: slice_exists = selected_key !== null && selected_key !== '';
	let rows_promise: any = null;
	$: {
		if (selected_key !== null && selected_key !== '') {
			rows_promise = fetchChunk({ df: df, keyidxs: [selected_key] });
		} else {
			rows_promise = null;
		}
	}

	let on_edit = async (event: any, column: string) => {
		if (on_change === null) return;

		dispatch(on_change.endpointId, {
			detail: {
				key: selected_key,
				column: column,
				value: event.detail.value
			}
		});
	};

	const format_number = (number: number) => {
		if (Number.isInteger(number) && number < 1000) {
			return number.toString();
		}
		const symbol_to_divisor = [
			['B', 9],
			['M', 6],
			['K', 3]
		];
		let number_str = number.toFixed(2);
		for (let i = 0; i < symbol_to_divisor.length; i++) {
			const symbol: string = symbol_to_divisor[i][0];
			const divisor: number = Math.pow(10, symbol_to_divisor[i][1]);
			if (number >= divisor) {
				number_str = `${(number / divisor).toFixed(2)}${symbol}`;
				break;
			}
		}
		return number_str;
	};
</script>

<div class="bg-slate-100 py-3 px-2 rounded-lg flex flex-col space-y-1">
	<div class="grid grid-cols-[1fr_auto] w-full">
		<div class="font-bold text-xl text-slate-600 self-start pl-2">
			{title}
		</div>

		<button
			class="self-end flex-auto rounded-md text-slate-100 px-1 flex items-center font-bold"
			class:bg-violet-500={!slice_exists}
			class:bg-slate-300={slice_exists}
			disabled={slice_exists}
			on:click={() => dispatch(on_slice_creation.endpointId, { detail: {} })}
		>
			<Scissors />
			Create Slice
		</button>
	</div>
	{#await schema_promise}
		Loading schema...
	{:then schema}
		{#each columns as column}
			<div class="">
				<div class="text-gray-600 font-mono">
					{column}
				</div>
				<div class="w-full flex">
					{#await rows_promise}
						Loading rows...
					{:then rows}
						{#if rows == null}
							<span class="italic">Unnamed slice</span>
						{:else}
							<Cell
								{...rows.get_cell(0, column)}
								on:edit={(event) => on_edit(event, column)}
								editable={true}
							/>
						{/if}
					{/await}
				</div>
			</div>
		{/each}
		<div class="m-2 flex flex-wrap justify-center gap-x-2 gap-y-2 pt-2">
			{#each Object.entries(stats) as [column, value]}
				<div class="bg-white rounded-md flex flex-col shadow-lg">
					<div class="text-slate-400 px-3 py-1 self-center">
						{#if rename[column]}
							{rename[column]}
						{:else}
							{column}
						{/if}
					</div>
					<div class="font-bold text-2xl px-3 self-center ">
						{#await rows_promise}
							Loading...
						{:then rows}
							{format_number(value)}
						{/await}
					</div>
				</div>
			{/each}
		</div>
	{/await}
</div>

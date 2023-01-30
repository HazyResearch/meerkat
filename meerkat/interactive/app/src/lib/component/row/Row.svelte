<script lang="ts">
	import type { DataFrameRef } from '$lib/api/dataframe';
	import BasicCell from '$lib/component/scalar/Scalar.svelte';
	import Cell from '$lib/shared/cell/Cell.svelte';
	import type { Endpoint } from '$lib/utils/types';
	import { getContext } from 'svelte';

	const { fetch_chunk, fetch_schema, dispatch } = getContext('Meerkat');

	export let df: DataFrameRef;
	export let selected_key: string;
	export let cell_specs: any;
	export let title: string = '';

	export let on_change: Endpoint = null;

	$: schema_promise = fetch_schema({ df: df });

	let rows_promise: any = null;
	$: {
		if (selected_key !== null && selected_key !== '') {
			rows_promise = fetch_chunk({ df: df, keyidxs: [selected_key] });
		} else {
			rows_promise = null;
		}
	}

	let on_edit = async (event: any, column: string) => {
		if (on_change === null) {
			return;
		}

		const promise = dispatch(on_change.endpoint_id, {
			key: selected_key,
			column: column,
			value: event.detail.value
		});
		promise.catch((error: TypeError) => {
			status = 'error';
			console.log(error);
		});
	};
</script>

<div class="bg-slate-100 py-3 px-2 rounded-lg drop-shadow-md flex flex-col space-y-1">
	{#if title != ''}
		<div class="font-bold text-xl text-slate-600 self-start pl-2">
			{title}
		</div>
	{/if}
	{#await schema_promise}
		Loading schema...
	{:then schema}
		{#each schema.columns as column, column_idx}
			{#if cell_specs[column.name] && cell_specs[column.name]['type'] !== 'stat'}
				<div class="">
					<div class="text-gray-600 font-mono">
						{column.name}
					</div>
					<div class="w-full flex">
						{#await rows_promise}
							Loading rows...
						{:then rows}
							{#if rows == null}
								No selection
							{:else}
								<Cell
									data={rows.rows[0][column_idx]}
									cell_component={column.cell_component}
									cell_props={column.cell_props}
									editable={cell_specs[column.name].type === 'editable'}
									on:edit={(event) => on_edit(event, column.name)}
								/>
							{/if}
						{/await}
					</div>
				</div>
			{/if}
		{/each}
		<div class="m-2 flex flex-wrap justify-center gap-x-2 gap-y-2 pt-2">
			{#each schema.columns as column, column_idx}
				{#if cell_specs[column.name] && cell_specs[column.name].type === 'stat'}
					<div class="bg-white rounded-md flex flex-col shadow-lg">
						<div class="text-slate-400 px-3 py-1 self-center">
							{#if cell_specs[column.name].name}
								{cell_specs[column.name].name}
							{:else}
								{column.name}
							{/if}
						</div>
						<div class="font-bold text-2xl px-3 self-center ">
							{#await rows_promise}
								Loading...
							{:then rows}
								{#if rows == null}
									No selection
								{:else}
									<BasicCell
										data={rows.rows[0][column_idx]}
										{...column.cell_props}
										percentage={false}
									/>
								{/if}
							{/await}
						</div>
					</div>
				{/if}
			{/each}
		</div>
	{/await}
</div>

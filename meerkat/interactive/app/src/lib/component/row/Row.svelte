<script lang="ts">
	import type { Writable } from 'svelte/store';
	import { getContext } from 'svelte';
	import type { DataFrameBox, EditTarget, Endpoint } from '$lib/utils/types';
	import { get } from 'svelte/store';
	import Cell from '$lib/shared/cell/Cell.svelte';
	import BasicCell from '$lib/shared/cell/basic/Basic.svelte';

	const { get_rows, get_schema, dispatch } = getContext('Interface');

	export let df: Writable<DataFrameBox>;
	export let primary_key_column: Writable<string>;
	export let selected_key: Writable<string>;
	export let cell_specs: any;
	export let title: Writable<string> = "";

	export let on_change: Endpoint = null;

	$: schema_promise = $get_schema($df.ref_id);

	let rows_promise: any = null;
	$: {
		console.log("selected key", $selected_key);
		if ($selected_key !== null && $selected_key !== "") {
			rows_promise = $get_rows($df.ref_id, null, null, null, null, $primary_key_column, [$selected_key]);
		} else {
			rows_promise = null;
		}
	}

	let on_edit = async (event: any, column: string) => {
		if (on_change === null) {
			return;
		}

		const promise = $dispatch(
			on_change.endpoint_id,
			{
				"key": $selected_key,
				"column": column,
				"value": event.detail.value
			}
		);
		promise.catch((error: TypeError) => {
			status = 'error';
			console.log(error);
		});
	};

	console.log("cell_specs", cell_specs);
	console.log("cell_specs", $cell_specs["name"].type);
	console.log("selected_key", $selected_key);

</script>

<div class="bg-slate-100 py-3 px-2 rounded-lg drop-shadow-md flex flex-col space-y-1">
	{#if $title != ""}
	<div class="font-bold text-xl text-slate-600 self-start pl-2">
		{$title}
	</div>
	{/if}
	{#await schema_promise}
		Loading schema...
	{:then schema}
		{#each schema.columns as column, column_idx}
        
			{#if $cell_specs[column.name] && $cell_specs[column.name]["type"] !== 'stat'}
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
									editable={$cell_specs[column.name].type === 'editable'}
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
				{#if $cell_specs[column.name] && $cell_specs[column.name].type === 'stat'}
					<div class="bg-white rounded-md flex flex-col shadow-lg">
						<div class="text-slate-400 px-3 py-1 self-center">
							{#if $cell_specs[column.name].name}
								{$cell_specs[column.name].name}
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
									<BasicCell data={rows.rows[0][column_idx]} {...column.cell_props} percentage={false}/>
								{/if}
							{/await}
						</div>
					</div>
				{/if}
			{/each}
		</div>
	{/await}
</div>

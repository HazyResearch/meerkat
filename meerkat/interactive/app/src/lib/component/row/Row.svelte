<script lang="ts">
	import type { Writable } from 'svelte/store';
	import { getContext } from 'svelte';
	import type { DataPanelBox, EditTarget } from '$lib/utils/types';
	import { get } from 'svelte/store';
	import Cell from '$lib/components/cell/Cell.svelte';
	import BasicCell from '$lib/components/cell/basic/Basic.svelte';

	const { get_rows, get_schema, edit_target } = getContext('Interface');

	export let dp: Writable<DataPanelBox>;
	export let idx: Writable<number>;
	export let target: EditTarget;
	export let cell_specs: Any = {};
	export let title: string = "";

	$: schema_promise = $get_schema($dp.box_id);
       

	$: target.target = get(target.target);

	let rows_promise: any = null;
	$: {
		if ($idx !== null) {
			rows_promise = $get_rows($dp.box_id, $idx, $idx + 1);
		} else {
			rows_promise = null;
		}
	}

	let on_edit = async (event: any, column: string) => {
		let modifications_promise = $edit_target(
			$dp.box_id,
			target,
			event.detail.value,
			column,
			[$idx],
			null,
			null
		);
		modifications_promise.catch((error: TypeError) => {
			status = 'error';
			console.log(error);
		});
	};
    console.log(cell_specs)

</script>

<div class="bg-slate-100 py-3 px-2 rounded-lg drop-shadow-md flex flex-col space-y-1">
	{#if title != ""}
	<div class="font-bold text-xl text-slate-600 self-start pl-2">
		{title}
	</div>
	{/if}
	{#await schema_promise}
		Loading...
	{:then schema}
		{#each schema.columns as column, column_idx}
        
			{#if cell_specs[column.name].type !== 'stat'}
				<div class="">
					<div class="text-gray-600 font-mono">
						{column.name}
					</div>
					<div class="w-full flex">
						{#await rows_promise}
							Loading...
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
				{#if cell_specs[column.name].type === 'stat'}
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

<script lang="ts">
	import type { Writable } from 'svelte/store';
	import { getContext } from 'svelte';
	import type { DataPanelBox, EditTarget } from '$lib/utils/types';
	import { get } from 'svelte/store';
	import Cell from '$lib/components/cell/Cell.svelte';

	const { get_rows, get_schema, edit_target } = getContext('Interface');

	export let dp: Writable<DataPanelBox>;
	export let idx: Writable<number>;
	export let target: EditTarget;

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
        modifications_promise
			.catch((error: TypeError) => {
				status = 'error';
				console.log(error);
			});
    }
</script>

<div class="bg-slate-100 py-3 px-2 rounded-lg drop-shadow-md flex flex-col space-y-1">
	<div class="font-bold text-xl text-slate-600 self-center justify-self-center">
		<!-- TODO(Sabri): This should be a customizable name in the future. -->
		Active Slice
	</div>
	{#await schema_promise}
		Loading...
	{:then schema}
		{#each schema.columns as column, column_idx}
			<div class="flex">
				<div class="text-gray-600 font-mono">
					{column.name}
				</div>
				<div>
					{#await rows_promise}
						Loading...
					{:then rows}
						<Cell
							data={rows.rows[0][column_idx]}
							cell_component={column.cell_component}
							cell_props={column.cell_props}
							editable={true}
                            on:edit={event => on_edit(event, column.name)}
						/>
					{/await}
				</div>
			</div>
		{/each}
	{/await}
</div>

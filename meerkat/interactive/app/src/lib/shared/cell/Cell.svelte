<script lang="ts">
	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';
	import type { CellInfo } from '$lib/utils/dataframe';
	import { createEventDispatcher } from 'svelte';
	import { setContext } from 'svelte/internal';
	import { writable } from 'svelte/store';

	export let data: any;
	export let cellInfo: CellInfo;
	export let cellComponent: string = '';
	export let cellProps: object = {};
	export let cellDataProp: string = 'data';
	export let editable: boolean = false;
	export let minWidth: number = 0;
	export let minHeight: number = 0;

	const dispatch = createEventDispatcher();
	setContext('cellEdit', (data: any) => {
		dispatch('edit', { value: data });
	});

	let cellInfoObj = {
		...cellInfo,
		style: { minWidth: writable(minWidth), minHeight: writable(minHeight) }
	};
	$: {
		cellInfoObj.style.minWidth.set(minWidth);
		cellInfoObj.style.minHeight.set(minHeight);
		console.log('minWidth:', minWidth, 'minHeight:', minHeight);
	}
	setContext('cellInfo', cellInfoObj);

	// need to actually create a new object, since we don't want to modify the
	// cell_props that were passed in
	$: {
		cellProps = { ...cellProps };
		cellProps[cellDataProp] = data;
		cellProps['cell_info'] = cellInfo;

		// iterate over cell_props and turn them into stores if they aren't already
		for (const [key, value] of Object.entries(cellProps)) {
			if (
				value !== undefined &&
				(value === null || value.subscribe === undefined) &&
				!key.startsWith('on_')
			) {
				cellProps[key] = writable(value);
			}
		}
	}
</script>

<DynamicComponent
	name={cellComponent}
	props={{
		...cellProps,
		editable: writable(editable)
	}}
/>

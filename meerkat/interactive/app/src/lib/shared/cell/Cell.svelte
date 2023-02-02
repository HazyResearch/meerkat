<script lang="ts">
	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';
	import { writable } from 'svelte/store';
	import { setContext } from 'svelte/internal';
	import {createEventDispatcher} from 'svelte';

	export let data: any;
	export let cell_component: string = '';
	export let cell_props: object = {};
	export let cell_data_prop: string = 'data';
	export let editable: boolean = false;

	const dispatch = createEventDispatcher();
	setContext('cellEdit', (data: any) => {
		dispatch('edit', {value: data});
	});

	// need to actually create a new object, since we don't want to modify the
	// cell_props that were passed in
	$: {
		cell_props = { ...cell_props };
		cell_props[cell_data_prop] = data;

		// iterate over cell_props and turn them into stores if they aren't already
		for (const [key, value] of Object.entries(cell_props)) {
			if (value === null || value.subscribe === undefined) {
				cell_props[key] = writable(value);
			}
		}
	}
</script>

<DynamicComponent name={cell_component} props={{...cell_props, editable: writable(editable)}} />

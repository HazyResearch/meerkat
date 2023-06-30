<script lang="ts">
	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';
	import type { CellInfo } from '$lib/utils/dataframe';
	import { createEventDispatcher } from 'svelte';
	import { setContext } from 'svelte/internal';
	import { writable } from 'svelte/store';

	export let name: string;
	export let props: object = {};

	// need to actually create a new object, since we don't want to modify the
	// cell_props that were passed in
	$: {
		props = { ...props };

		// iterate over cell_props and turn them into stores if they aren't already
		for (const [key, value] of Object.entries(props)) {
			if (
				value !== undefined &&
				(value === null || value.subscribe === undefined) &&
				!key.startsWith('on_')
			) {
				props[key] = writable(value);
			}
		}
	}

</script>

<DynamicComponent name={name} props={props} />

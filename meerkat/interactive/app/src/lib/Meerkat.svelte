<script lang="ts">
	import {
		add,
		aggregate_sliceby,
		dispatch,
		edit,
		edit_target,
		get_rows,
		get_schema,
		get_sliceby_info,
		get_sliceby_rows,
		match,
		remove_row_by_index,
		store_trigger
	} from './api/api';
	import { setContext } from 'svelte/internal';

	export const context = {
		add: add,
		aggregate_sliceby: aggregate_sliceby,
		dispatch: dispatch,
		edit: edit,
		edit_target: edit_target,
		get_rows: get_rows,
		get_schema: get_schema,
		get_sliceby_info: get_sliceby_info,
		get_sliceby_rows: get_sliceby_rows,
		match: match,
		remove_row_by_index: remove_row_by_index,
		store_trigger: store_trigger
	};

	$: {
		setContext('Meerkat', context);
	}
</script>

<slot />

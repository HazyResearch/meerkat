<script lang="ts">
	import {
		add,
		aggregate_sliceby,
		dispatch,
		fetch_chunk, 
		fetch_schema,
		get_sliceby_info,
		get_sliceby_rows,
		match,
		remove_row_by_index,
		update_store
	} from './api/api';
	import { setContext } from 'svelte/internal';

	export const context = {
		add: add,
		aggregate_sliceby: aggregate_sliceby,
		dispatch: dispatch,
		fetch_chunk: fetch_chunk,
		fetch_schema: fetch_schema,
		get_sliceby_info: get_sliceby_info,
		get_sliceby_rows: get_sliceby_rows,
		match: match,
		remove_row_by_index: remove_row_by_index,
		update_store: update_store
	};

	$: {
		setContext('Meerkat', context);
	}
</script>

<slot />

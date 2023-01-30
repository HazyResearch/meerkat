<script lang="ts">
	import { setContext } from 'svelte/internal';
	import { dispatch, fetch_chunk, fetch_schema, update_store } from './utils/api';

	setContext('Meerkat', {
		dispatch: dispatch,
		fetch_chunk: fetch_chunk,
		fetch_schema: fetch_schema,
		update_store: update_store
	});
</script>

<slot />

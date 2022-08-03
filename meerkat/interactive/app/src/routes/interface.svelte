<script context="module">
	 
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		const id = url.searchParams.get('id') || '/dashboard';
		// Need to fix this to use the correct api_url 
		const api_url = 'http://127.0.0.1:7861/interface/' + id + '/config';
		const response = await fetch(api_url);

		return {
			status: response.status,
			props: {
				config: response.ok && (await response.json()),
				id: id
			}
		};
	}
</script>

<script lang="ts">
	import TableView from '$lib/TableView.svelte';
	import _, { unzip } from 'underscore';

	export let config: any;
</script>

<div>
	{#if config.type == 'table'}
		<TableView nrows={config.nrows} dp={config.dp} />
	{:else}
		<div>Type not recognized.</div>
	{/if}
</div>
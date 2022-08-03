<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		const id = url.searchParams.get('id');
		const response = await fetch(`${get(api_url)}/interface/` + id + '/config');

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
	import { get } from 'svelte/store';
	import { api_url } from './network/stores';

	export let config: any;
</script>

<div>
	{#if config.type == 'table'}
		<TableView nrows={config.nrows} dp={config.dp} />
	{:else}
		<div>Type not recognized.</div>
	{/if}
</div>

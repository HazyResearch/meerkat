<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		const id = url.searchParams.get('id');
		const response = await fetch(`${get(api_url)}/interface/${id}/config`);

		return {
			status: response.status,
			props: {
				config: response.ok && (await response.json()),
			}
		};
	}
</script>

<script lang="ts">
	import TableView from '$lib/TableView.svelte';
	import { get } from 'svelte/store';
	import { api_url } from './network/stores';
	import Prism from "../lib/components/cell/code/Code.svelte";


	export let config: any;



</script>

<div class="h-[800px]">
	{#if config.type == 'table'}
		<TableView nrows={config.nrows} datapanel_id={config.dp} />
	{:else}
		<div>Type not recognized.</div>
	{/if}
</div>

<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		const id = url.searchParams.get('id');
		const response = await fetch(`${get(api_url)}/interface/${id}/config`);

		return {
			status: response.status,
			props: {
				config: response.ok && (await response.json())
			}
		};
	}
</script>

<script lang="ts">
	import TableView from '$lib/TableView.svelte';
	import SliceCards from '$lib/components/sliceby/SliceCards.svelte';
	import { get } from 'svelte/store';
	import { api_url } from './network/stores';
	import Prism from '../lib/components/cell/code/Code.svelte';

	export let config: any;
</script>

<div class="h-[800px]">
	{#if config.component === 'table'}
		<TableView nrows={config.props.nrows} datapanel_id={config.props.dp} />
	{:else if config.component === 'sliceby-cards'}
		<SliceCards {...config.props} />
	{:else}
		<div>Type not recognized.</div>
	{/if}
</div>

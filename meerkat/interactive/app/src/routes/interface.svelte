<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		console.log("In here.");
		const id = url.searchParams.get('id');
		const response = await fetch(`${get(api_url)}/interface/${id}/config`);

		console.log(get(api_url));
		console.log(response);

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
import Everything from '$lib/components/blocks/Everything.svelte';

	export let config: any;
	console.log(config);
</script>

<div class="h-[800px]">
	{#if config.component === 'table'}
		<Everything/>
		<!-- <TableView nrows={config.props.nrows} datapanel_id={config.props.dp} /> -->
	{:else if config.component === 'sliceby-cards'}
		<SliceCards {...config.props} />
	{:else}
		<div>Type not recognized.</div>
	{/if}
</div>

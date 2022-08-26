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
	import { get } from 'svelte/store';
	import { api_url } from '../network/stores';
    import Interface from '$lib/components/abstract/Interface.svelte'; 
    import Block from '$lib/components/abstract/block.svelte';
	
	export let config: any;
</script>

<Interface klass="h-[800px]">
    <Block base_datapanel_id={config.props.dp}>
        <TableView nrows={config.props.nrows} datapanel_id={config.props.dp} />
    </Block>
</Interface>

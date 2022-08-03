<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		const id = url.searchParams.get('id') || '/dashboard';

		const api_url = 'http://127.0.0.1:7860/interface/config?id=' + id;
		const response = await fetch(api_url);

		return {
			status: response.status,
			props: {
				config: response.ok && (await response.json()),
				id: id, 
			}
		};
	}
</script>

<script lang="ts">
	import TableView from '$lib/TableView.svelte';
	import _, { unzip } from "underscore";
	import { api_url } from './network/stores';
	import { get, post } from '$lib/utils/requests';

	export let config: any;
	export let id: number;

	let columns: Array<string> = [];
	let types: Array<string> = [];
	let rows;
	let indices: Array<number> = [];

	let loader = async (start: number, end: number) => {
		let data_promise = await post(
			`${$api_url}/dp/${id}/rows`, 
			{start: start, end: end}
		);

		columns = data_promise.column_info.map((col: any) => col.name);
		rows = data_promise.rows;
		types = data_promise.column_info.map((col: any) => col.type);
		indices = data_promise.indices;
	}
	
	let data_promise = loader(0, 10);

</script>

<div>
	{#await data_promise}
		<TableView columns={["Loading..."]} rows={rows} types={types} loader={loader} nrows={config.params.nrows} />
	{:then data} 
		{#if config.type == 'table'}
			<TableView bind:columns={columns} bind:rows={rows} bind:types={types} loader={loader} nrows={config.params.nrows} />
		{:else}
			<div>Type not recognized.</div>
		{/if}
	{/await}
	
</div>

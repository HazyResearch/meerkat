<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		const id = url.searchParams.get('id') || '/dashboard';

		const api_url = 'http://127.0.0.1:7860/interface/config?id=' + id;
		const response = await fetch(api_url);

		return {
			status: response.status,
			props: {
				config: response.ok && (await response.json())
			}
		};
	}
</script>

<script lang="ts">
	import { dataset_dev } from 'svelte/internal';
	import Table  from '$lib/interfaces/Table.svelte';


	async function fetch_url(url: string): Promise<any> {
		const res: Response = await fetch(url);
		if (!res.ok) {
			throw new Error('HTTP status ' + res.status);
		}
		const json = await res.json();
		return json;
	}

	export let config: any;

	// GET request from the Python API to pull in some data (used below in HTML)
	// let data_promise = fetch_url("http://127.0.0.1:7860/config?id=123");
	// data_promise.then(data => console.log("JSON", data));

	// do stuff
</script>

<div>
	<!-- {#await data_promise}
		Loading data...
	{:then data} 
		{ data }
	{/await} -->
	{#if config.type == 'table'}
		<Table nrows={config.params.nrows} />
	{:else}
		<div>Type not recognized.</div>
	{/if}
</div>

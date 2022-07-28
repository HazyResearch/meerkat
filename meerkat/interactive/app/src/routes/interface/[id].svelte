<!-- <script context="module" lang="ts">
	export async function load({ url }: { url: URL }) {
		const id = url.searchParams.get('id') || '/dashboard';
		console.log("here")

		let getUrl = 'http://localhost:7860/interface/config?id=' + id
		// const response = await fetch('http://localhost:7860/interface/config?id=' + id);


		return {
			props: {
				getUrl: getUrl
			}
		};
	}
</script> -->

<script lang="ts">
	import { dataset_dev } from 'svelte/internal';

	export let getUrl: any = "test";
	export let id: any; 

	
	async function fetch_url (url: string): Promise<any> {
		const res: Response = await fetch(url);
		if (!res.ok) {
			throw new Error("HTTP status " + res.status);
		}
		const json = await res.json();
		return json;
	};

    // GET request from the Python API to pull in some data (used below in HTML)
	// let data_promise = fetch_url("http://127.0.0.1:7861/config?id=" + id);
	let data_promise = fetch_url("http://localhost:7860/test");
	data_promise.then(data => console.log("JSON", data));
	



	// do stuff
</script>

<div> { id } </div>

<div>
	{#await data_promise}
		Loading data {getUrl}
	{:then data} 
		{ data }
	{/await}
</div>

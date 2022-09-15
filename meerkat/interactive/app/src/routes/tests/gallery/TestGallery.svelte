<script lang="ts">
	import { get_rows } from '$lib/api/datapanel';

	import Gallery from '$lib/components/gallery/Cards.svelte';
	import { api_url } from '$network/stores';

	let column_infos: Array<any> = [];
	let rows: Array<any> = [];

	let loader = async (start: number, end: number) => {
		let data_promise = await get_rows($api_url, 'test-imagenette', start, end);
		column_infos = data_promise.column_infos;
		rows = data_promise.rows;
        return data_promise;
	};
	let data_promise = loader(0, 100);
</script>

{#await data_promise}
	Loading...
{:then data}
	<Gallery
		schema={{columns: column_infos}}
		rows={data}
		main_column={'img'}
		tag_columns={['label', 'split']}
	/>
{/await}

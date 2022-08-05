<script lang="ts">
    import Gallery from "$lib/components/gallery/Gallery.svelte";
    import { post } from '$lib/utils/requests';
	import { api_url } from '$network/stores';

    let columns: Array<string> = [];
    let rows: Array<any> = [];

    let loader = async (start: number, end: number) => {
        let data_promise = await post(
            `${$api_url}/dp/test-imagenette/rows`, 
            { start: start, end: end }
        );

        columns = data_promise.column_info.map((col: any) => col.name);
        rows = data_promise.rows;
    };
    let data_promise = loader(0, 100);

</script>


{#await data_promise}
    Loading...
{:then data} 
    <Gallery 
        columns={columns}
        rows={rows}
        main_column={"img"}
        tag_columns={["split", "img_path"]}
    />
{/await}


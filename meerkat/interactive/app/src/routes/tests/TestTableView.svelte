<script lang="ts">
    import TableView from "$lib/TableView.svelte";
    import { get } from "$lib/utils/requests";
    import { api_url } from "src/routes/network/stores";

    import _, { unzip } from "underscore";

    // TODO: replace with dummy rows from API
    let data_promise = get(`${$api_url}/dp/rows`);

    let parse_data = (json: any) => {
        return {
            columns: Object.keys(json.data),
            types: Object.values(json.types) as Array<string>,
            rows: unzip(Object.values(json.data))
        };
    };

</script>

{#await data_promise}
    <div/>
{:then json}
    <TableView {...parse_data(json)}/>
{/await}

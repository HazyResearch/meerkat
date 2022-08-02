<script lang="ts">
    import Table from "$lib/components/table/Table.svelte";

    import _, { unzip } from "underscore";

    async function fetch_url (url: string): Promise<any> {
		const res: Response = await fetch(url);
		if (!res.ok) {
			throw new Error("HTTP status " + res.status);
		}
		const json = await res.json();
		return json;
	};

    let data_promise = fetch_url(`http://localhost:7860/dp/rows`);

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
    <Table {...parse_data(json)}/>
{/await}

<script lang="ts">
    import { api_url } from '$network/stores.js';
    import { get_schema } from '$lib/api/datapanel';
    export let dp: Writable;


	let schema_promise = get_schema($api_url, $dp.box_id);
	dp.subscribe((value: any) => {
		schema_promise = get_schema($api_url, $dp.box_id);
	});
	// export let selection: any;
	// export let
</script>

<div class="bg-slate-100">
	Gallery
	{#await schema_promise}
		waiting....
	{:then schema}
		<div class="flex space-x-3">
			{#each schema.columns as column_info}
				<div class="bg-violet-200 rounded-md px-3 font-bold text-slate-700">
					{column_info.name}
				</div>
			{/each}
		</div>
	{:catch error}
		{error}
	{/await}
</div>

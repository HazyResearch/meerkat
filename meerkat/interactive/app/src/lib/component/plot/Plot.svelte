<script lang="ts">
	import { api_url } from '$network/stores.js';
    import { get, writable } from 'svelte/store';
	import type { Writable } from 'svelte/store';
	import { getContext } from 'svelte';

	const { get_schema } = getContext('Interface');


	export let dp: Writable;
	export let selection: Writable;
	export let x: Writable;
	export let y: Writable;
    export let x_label: Writable;
    export let y_label: Writable;
    export let type: string;

	$: schema_promise = $get_schema($dp.box_id);

	// export let selection: any;
	// export let
</script>

<div class="bg-slate-100">
	Plot of type {type}

    x: {$x}, y: {$y}
    x_label: {$x_label}, y_label: {$y_label}



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

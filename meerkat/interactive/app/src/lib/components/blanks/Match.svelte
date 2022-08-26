<script lang="ts">
    import { get, writable, type Writable } from 'svelte/store';
    import { get_schema } from '$lib/api/datapanel';
    import { global_stores } from '$lib/components/blanks/stores';
    import { post } from '$lib/utils/requests';
    import {getContext} from "svelte";
    
    const {schema, add} = getContext("Interface")

    export let dp: Writable;
    export let against: Writable<string>;
    export let col: Writable<string>;
    export let text: Writable<string>;

    $: schema_promise = $schema($dp.box_id);

	let on_add = async () => {
        let box_id = $dp.box_id
        let column_name =  `add(${$against})`;
        col.set(column_name)
        $add(box_id, column_name)
	};
</script>


<div class="w-full py-5 px-2 bg-slate-100 ">
    Match

    <input type="text" bind:value={$against}>
    <input type="text" bind:value={$text}>


    Column: {$col}

    <button class="bg-slate-500" on:click={on_add}> Add </button>
    
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
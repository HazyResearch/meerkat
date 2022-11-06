<script lang="ts">
    import { getContext } from "svelte";
    import { _data, _boxes } from './stores.js';

    const { get_box } = getContext('Block');

    export let df_box_id: string;
    
    // letValue: str;
    // _boxes[df_box_id].subscribe(
    //     value => {
    //         console.log(value);
    //         store_value = value;
    //     }
    // );

    // $: df_box_id_store = _boxes.get(df_box_id);
    $: df_promise = $get_box(df_box_id); //console.log(letValue)}

    // // Why is there a $ in front of get_box?
    // // neither get_box nor df_box_id are changing, so we wouldn't expected
    // // df_promise to be updated
    // $: df_promise = $get_box(df_box_id);

</script>

<div class="bg-slate-100">
    {#await df_promise}
        Waiting...
    {:then df}
        <!-- {df.column_infos} -->
        {df.rows.length}
    {:catch error}
        {error}
    {/await}
</div>

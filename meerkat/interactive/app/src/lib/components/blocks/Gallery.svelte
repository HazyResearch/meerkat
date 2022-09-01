<script lang="ts">
    import { getContext } from "svelte";
    import { _data, _boxes } from './stores.js';

    const { get_box } = getContext('Block');

    export let dp_box_id: string;
    
    // letValue: str;
    // _boxes[dp_box_id].subscribe(
    //     value => {
    //         console.log(value);
    //         store_value = value;
    //     }
    // );

    // $: dp_box_id_store = _boxes.get(dp_box_id);
    $: dp_promise = $get_box(dp_box_id); //console.log(letValue)}

    // // Why is there a $ in front of get_box?
    // // neither get_box nor dp_box_id are changing, so we wouldn't expected
    // // dp_promise to be updated
    // $: dp_promise = $get_box(dp_box_id);

</script>

<div class="bg-slate-100">
    {#await dp_promise}
        Waiting...
    {:then dp}
        <!-- {dp.column_infos} -->
        {dp.rows.length}
    {:catch error}
        {error}
    {/await}
</div>

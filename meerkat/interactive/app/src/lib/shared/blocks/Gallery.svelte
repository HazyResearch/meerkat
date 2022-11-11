<script lang="ts">
    import { getContext } from "svelte";
    import { _data, _refs } from './stores.js';

    const { get_ref } = getContext('Block');

    export let df_ref_id: string;
    
    // letValue: str;
    // _refs[df_ref_id].subscribe(
    //     value => {
    //         console.log(value);
    //         store_value = value;
    //     }
    // );

    // $: df_ref_id_store = _refs.get(df_ref_id);
    $: df_promise = $get_ref(df_ref_id); //console.log(letValue)}

    // // Why is there a $ in front of get_ref?
    // // neither get_ref nor df_ref_id are changing, so we wouldn't expected
    // // df_promise to be updated
    // $: df_promise = $get_ref(df_ref_id);

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

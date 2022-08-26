<script lang="ts">
    import { getContext } from "svelte";

    const { dispatch, datapanels_store, get_dp, get_dp_id } = getContext('Block');
    
    let alias;
    let action;
    
    let handle_click = (e) => {
        dispatch(action, alias, 9);
    };

</script>

<div class="text-violet-400">
    <input placeholder="dp_alias" bind:value={alias}/>
    <input placeholder="action" bind:value={action}/>
    {alias} {action}
    <button on:click={handle_click}>Dispatch</button>
</div>
<div class="text-violet-400 mb-8">
    <!-- <div>dp_a: {$datapanels_store.dp_a.data.q1}</div>
    <div>dp_b: {$datapanels_store.dp_b.data.q1}</div>
    <div>dp_c: {$datapanels_store.dp_c.data.q1}</div>
    <div>dp_c: {$get_dp('dp_c').data.q1}</div>
    <div>dp_c: {$get_dp_id('dp_c')}</div>
    <div>dp_d: {$datapanels_store.dp_d.data.q1}</div> -->
    <div>dp_a: {$get_dp('dp_a', 'q1')}</div>
    <div>dp_b: {$get_dp('dp_b', 'q1')}</div>
    <div>dp_c: {$get_dp('dp_c', 'q1')}</div>
    <div>dp_d: {$get_dp('dp_d', 'q1')}</div>
</div>

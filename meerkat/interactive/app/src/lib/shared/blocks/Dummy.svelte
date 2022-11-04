<script lang="ts">
    import { getContext } from "svelte";

    const { dispatch, dataframes_store, get_df, get_df_id } = getContext('Block');
    
    let alias;
    let action;
    
    let handle_click = (e) => {
        dispatch(action, alias, 9);
    };

</script>

<div class="text-violet-400">
    <input placeholder="df_alias" bind:value={alias}/>
    <input placeholder="action" bind:value={action}/>
    {alias} {action}
    <button on:click={handle_click}>Dispatch</button>
</div>
<div class="text-violet-400 mb-8">
    <!-- <div>df_a: {$dataframes_store.df_a.data.q1}</div>
    <div>df_b: {$dataframes_store.df_b.data.q1}</div>
    <div>df_c: {$dataframes_store.df_c.data.q1}</div>
    <div>df_c: {$get_df('df_c').data.q1}</div>
    <div>df_c: {$get_df_id('df_c')}</div>
    <div>df_d: {$dataframes_store.df_d.data.q1}</div> -->
    <div>df_a: {$get_df('df_a', 'q1')}</div>
    <div>df_b: {$get_df('df_b', 'q1')}</div>
    <div>df_c: {$get_df('df_c', 'q1')}</div>
    <div>df_d: {$get_df('df_d', 'q1')}</div>
</div>

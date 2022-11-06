<script lang="ts">
    import { getContext } from "svelte";

    const { dispatch, get_df } = getContext('Block');
    
    let alias;
    let action;
    let args = 9;

    export let count = 1;
    
    let handle_click = (e) => {
        dispatch(action, alias, args);
    };

</script>

<div class="text-violet-400">
    DummyTable | Count = {count} |
    <button on:click={() => count += 1}>Count++</button>
</div>
<div class="text-violet-400">
    <input placeholder="df_alias" bind:value={alias}/>
    <input placeholder="action" bind:value={action}/>
    <input placeholder="args" bind:value={args}/>
    {alias} {action} {args}
    <button on:click={handle_click}>Dispatch</button>
</div>
<div class="text-violet-400 mb-8">
    <div>df: {$get_df('df', `q${count % 2 + 1}`)}</div>
</div>

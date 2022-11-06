<script lang="ts">
    import { setContext } from "svelte";
    import { writable } from "svelte/store";
    import { _data } from './stores.js';

    // DataFrames that will be used by shared in this block
    // Should be an object, mapping 
    // user-defined (front-end only) arbitrary names
    // to objects with the following properties:
    // - id: a unique identifier for the dataframe
    // - data: an array of previously executed get_rows queries
    export let dataframes: any;

    // TODO: how should information about active be handled?
    export let active: { [key: string]: any; } = {};

    // Use a store with setContext so that the context is reactive
    let _dataframes = writable(dataframes);

    // Bind the store value to the prop as a reactive statement
    // This is necessary for the store to react to changes in the prop
    // This is also how LayerCake uses stores with setContext
    $: $_dataframes = dataframes;

    // The dispatch function will make all calls to the backend
    // (called `actions`)
    // TODO: add an object containing all possible dispatchable actions
    let dispatch = (action, alias, args) => {
        console.log("dispatch", action, alias, args);
        
        // Don't dispatch if the dataframe is view only
        if ($_dataframes[alias].view_only) {
            // Dispatch is a no-op
            return;
        }

        // TODO: normally `action` should call an operation
        if (action === 'q1' || action === 'q2') {
            $_data[$_dataframes[alias].id][action].push(args);
        } else if (action === 'switch') {
            console.log("here");
            $_dataframes[alias].id = args;
        }
        console.log($_dataframes);
        dataframes = dataframes;
    };

    // A reactive function declaration, so that the function body is 
    // reactive to changes in the _dataframes store
    $: get_df = (alias: string, args: string)  => {
        console.log("get_df", alias, args);
        // TODO: if isolated, then return the isolated dataframe
        
        // Set the active prop
        // TODO: make this more general
        active[alias] = args;
        // console.log(active);

        // Map alias -> id, then use id to look up data, then
        // args should help run an action like get_rows
        return $_data[$_dataframes[alias].id][args];
    };
    // Use a store with setContext so that the context is reactive
    const _get_df = writable(get_df);
    // Bind the store value to the function as a reactive statement to keep it in sync
    $: $_get_df = get_df;


    $: get_df_id = (alias: string) => {
        // TODO: if isolated, then return the id of the isolated dataframe
        return $_dataframes[alias].id;
    };
    // Use a store with setContext so that the context is reactive
    const _get_df_id = writable(get_df_id);
    // Bind the store value to the function as a reactive statement to keep it in sync
    $: $_get_df_id = get_df_id;

    // The Block provides its shared some common functionality
    $: context = {
        dispatch: dispatch,
        dataframes_store: _dataframes,
        get_df: _get_df,
        get_df_id: _get_df_id,
    };
    $: setContext("Block", context);
</script>

<!-- Block doesn't handle any layout or styling. -->
<slot/>
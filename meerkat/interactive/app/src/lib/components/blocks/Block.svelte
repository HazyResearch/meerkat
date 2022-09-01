<script lang="ts">
    import { setContext } from "svelte";
    import { writable } from "svelte/store";
    import { _data } from './stores.js';

    // DataPanels that will be used by components in this block
    // Should be an object, mapping 
    // user-defined (front-end only) arbitrary names
    // to objects with the following properties:
    // - id: a unique identifier for the datapanel
    // - data: an array of previously executed get_rows queries
    export let datapanels: any;

    // TODO: how should information about active be handled?
    export let active: { [key: string]: any; } = {};

    // Use a store with setContext so that the context is reactive
    let _datapanels = writable(datapanels);

    // Bind the store value to the prop as a reactive statement
    // This is necessary for the store to react to changes in the prop
    // This is also how LayerCake uses stores with setContext
    $: $_datapanels = datapanels;

    // The dispatch function will make all calls to the backend
    // (called `actions`)
    // TODO: add an object containing all possible dispatchable actions
    let dispatch = (action, alias, args) => {
        console.log("dispatch", action, alias, args);
        
        // Don't dispatch if the datapanel is view only
        if ($_datapanels[alias].view_only) {
            // Dispatch is a no-op
            return;
        }

        // TODO: normally `action` should call an operation
        if (action === 'q1' || action === 'q2') {
            $_data[$_datapanels[alias].id][action].push(args);
        } else if (action === 'switch') {
            console.log("here");
            $_datapanels[alias].id = args;
        }
        console.log($_datapanels);
        datapanels = datapanels;
    };

    // A reactive function declaration, so that the function body is 
    // reactive to changes in the _datapanels store
    $: get_dp = (alias: string, args: string)  => {
        console.log("get_dp", alias, args);
        // TODO: if isolated, then return the isolated datapanel
        
        // Set the active prop
        // TODO: make this more general
        active[alias] = args;
        // console.log(active);

        // Map alias -> id, then use id to look up data, then
        // args should help run an action like get_rows
        return $_data[$_datapanels[alias].id][args];
    };
    // Use a store with setContext so that the context is reactive
    const _get_dp = writable(get_dp);
    // Bind the store value to the function as a reactive statement to keep it in sync
    $: $_get_dp = get_dp;


    $: get_dp_id = (alias: string) => {
        // TODO: if isolated, then return the id of the isolated datapanel
        return $_datapanels[alias].id;
    };
    // Use a store with setContext so that the context is reactive
    const _get_dp_id = writable(get_dp_id);
    // Bind the store value to the function as a reactive statement to keep it in sync
    $: $_get_dp_id = get_dp_id;

    // The Block provides its components some common functionality
    $: context = {
        dispatch: dispatch,
        datapanels_store: _datapanels,
        get_dp: _get_dp,
        get_dp_id: _get_dp_id,
    };
    $: setContext("Block", context);
</script>

<!-- Block doesn't handle any layout or styling. -->
<slot/>
<script lang="ts">
import { get_rows } from "$lib/api/dataframe.js";

    import { setContext } from "svelte";
    import { writable } from "svelte/store";
    import { filter_ref, undo_ref } from "$lib/api/dataframe";
    import { api_url } from '$lib/../routes/network/stores.js';

    import { _data, _refs } from './stores.js';

    // export let alias_to_ref_id = new Map();

    // DataFrames that will be used by shared in this block
    // Should be an object, mapping 
    // user-defined (front-end only) arbitrary names
    // to objects with the following properties:
    // - id: a unique identifier for the dataframe
    // - data: an array of previously executed get_rows queries
    // export let dataframes: any;

    // TODO: how should information about active be handled?
    // export let active: { [key: string]: any; } = {};

    // Use a store with setContext so that the context is reactive
    // let _dataframes = writable(dataframes);

    // Bind the store value to the prop as a reactive statement
    // This is necessary for the store to react to changes in the prop
    // This is also how LayerCake uses stores with setContext
    // $: $_dataframes = dataframes;


    // The dispatch function will make all calls to the backend
    // (called `actions`)
    // TODO: add an object containing all possible dispatchable actions
    let dispatch = async (action, ref_id, args) => {
        console.log("dispatch", action, ref_id, args);
        let update;
        if (action === 'filter') {
            // Set of updated ref ids are returned by teh backend
            update = await filter_ref($api_url, ref_id, args);
            console.log(update)
        } else if (action === 'unfilter') {
            // Set of updated ref ids are returned by teh backend
            update = await undo_ref($api_url, ref_id, args);
        }
        console.log(_refs);
        let ref_store = _refs[update.ref_id];
        console.log(ref_store);
        ref_store.set(update.op_id);
        console.log("Went through dispatch");
        console.log(get(ref_store))

        return {
            'undo': () => {
                dispatch('unfilter', update.ref_id, update.op_id)
            }
        }
    };

    // let dispatch = (action, alias, args) => {
    //     console.log("dispatch", action, alias, args);

    //     const ref_id = alias_to_ref_id.get(alias);

    //     if (action === 'filter') {
    //         // Set of updated ref ids are returned by teh backend
    //         let updated_refs = filter(ref_id, args);
    //     } else if (action === 'unfilter') {
    //         // Set of updated ref ids are returned by teh backend
    //         let updated_refs = unfilter(ref_id, args);
    //     }
    // };

    let store_value;
    // let subscribed_refs = new Set();
    // // subscribed_refs.add(ref_id);
    
    _refs['ref_1'].subscribe(
        value => {
            console.log("VALUE", value);
            store_value = value;
        }
    );
    
    
    import { get } from 'svelte/store';
    $: get_ref = async (ref_id: string, args: string)  => {
    
        // Get the ref store
        // let ref_store = _refs.get(ref_id);
        // let _ref_id = get(ref_store);

        console.log("get_ref", ref_id, args, ref_id, $api_url);
        store_value = store_value;

        // Get the data for the ref
        // args should help run an action like get_rows
        let return_value = await get_rows($api_url, ref_id, 0, 10);
        console.log("return_value", return_value);
        return return_value;
    };

    // // A reactive function declaration, so that the function body is 
    // // reactive to changes in the _dataframes store
    // $: get_ref = (alias: string, args: string)  => {
    //     console.log("get_ref", alias, args);
    //     // Map alias -> ref id
    //     let ref_id = alias_to_ref_id.get(alias);

    //     // Get the ref store
    //     let ref_store = $_refs.get(ref_id);
    //     let _ref_id = $ref_store;

    //     // Get the data for the ref
    //     // args should help run an action like get_rows
    //     return get_rows(_ref_id, args);
    // };
    // Use a store with setContext so that the context is reactive
    const _get_ref = writable(get_ref);
    // Bind the store value to the function as a reactive statement to keep it in sync
    $: $_get_ref = get_ref;

    // The Block provides its shared some common functionality
    $: context = {
        dispatch: dispatch,
        get_ref: _get_ref,
    };
    $: setContext("Block", context);
</script>

<!-- Block doesn't handle any layout or styling. -->
<slot/>
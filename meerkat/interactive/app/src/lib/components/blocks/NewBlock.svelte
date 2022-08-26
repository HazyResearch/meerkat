<script lang="ts">
import { get_rows } from "$lib/api/datapanel.js";

    import { setContext } from "svelte";
    import { writable } from "svelte/store";
    import { filter_box, undo_box } from "$lib/api/datapanel";
    import { api_url } from '$lib/../routes/network/stores.js';

    import { _data, _boxes } from './stores.js';

    // export let alias_to_box_id = new Map();

    // DataPanels that will be used by components in this block
    // Should be an object, mapping 
    // user-defined (front-end only) arbitrary names
    // to objects with the following properties:
    // - id: a unique identifier for the datapanel
    // - data: an array of previously executed get_rows queries
    // export let datapanels: any;

    // TODO: how should information about active be handled?
    // export let active: { [key: string]: any; } = {};

    // Use a store with setContext so that the context is reactive
    // let _datapanels = writable(datapanels);

    // Bind the store value to the prop as a reactive statement
    // This is necessary for the store to react to changes in the prop
    // This is also how LayerCake uses stores with setContext
    // $: $_datapanels = datapanels;


    // The dispatch function will make all calls to the backend
    // (called `actions`)
    // TODO: add an object containing all possible dispatchable actions
    let dispatch = async (action, box_id, args) => {
        console.log("dispatch", action, box_id, args);
        let update;
        if (action === 'filter') {
            // Set of updated box ids are returned by teh backend
            update = await filter_box($api_url, box_id, args);
            console.log(update)
        } else if (action === 'unfilter') {
            // Set of updated box ids are returned by teh backend
            update = await undo_box($api_url, box_id, args);
        }
        console.log(_boxes);
        let box_store = _boxes[update.box_id];
        console.log(box_store);
        box_store.set(update.op_id);
        console.log("Went through dispatch");
        console.log(get(box_store))

        return {
            'undo': () => {
                dispatch('unfilter', update.box_id, update.op_id)
            }
        }
    };

    // let dispatch = (action, alias, args) => {
    //     console.log("dispatch", action, alias, args);

    //     const box_id = alias_to_box_id.get(alias);

    //     if (action === 'filter') {
    //         // Set of updated box ids are returned by teh backend
    //         let updated_boxes = filter(box_id, args);
    //     } else if (action === 'unfilter') {
    //         // Set of updated box ids are returned by teh backend
    //         let updated_boxes = unfilter(box_id, args);
    //     }
    // };

    let store_value;
    // let subscribed_boxes = new Set();
    // // subscribed_boxes.add(box_id);
    
    _boxes['box_1'].subscribe(
        value => {
            console.log("VALUE", value);
            store_value = value;
        }
    );
    
    
    import { get } from 'svelte/store';
    $: get_box = async (box_id: string, args: string)  => {
    
        // Get the box store
        // let box_store = _boxes.get(box_id);
        // let _box_id = get(box_store);

        console.log("get_box", box_id, args, box_id, $api_url);
        store_value = store_value;

        // Get the data for the box
        // args should help run an action like get_rows
        let return_value = await get_rows($api_url, box_id, 0, 10);
        console.log("return_value", return_value);
        return return_value;
    };

    // // A reactive function declaration, so that the function body is 
    // // reactive to changes in the _datapanels store
    // $: get_box = (alias: string, args: string)  => {
    //     console.log("get_box", alias, args);
    //     // Map alias -> box id
    //     let box_id = alias_to_box_id.get(alias);

    //     // Get the box store
    //     let box_store = $_boxes.get(box_id);
    //     let _box_id = $box_store;

    //     // Get the data for the box
    //     // args should help run an action like get_rows
    //     return get_rows(_box_id, args);
    // };
    // Use a store with setContext so that the context is reactive
    const _get_box = writable(get_box);
    // Bind the store value to the function as a reactive statement to keep it in sync
    $: $_get_box = get_box;

    // The Block provides its components some common functionality
    $: context = {
        dispatch: dispatch,
        get_box: _get_box,
    };
    $: setContext("Block", context);
</script>

<!-- Block doesn't handle any layout or styling. -->
<slot/>
<!-- 
    Block
    - Linked to a single DataPanel. 
    - Manages all API requests.
    - Provides important state variables to child components.
    - Connected to other Blocks via a BlockGroup.
    - If any Block in the BlockGroup is updated, then all the Blocks connected to the BlockGroup are triggered.

 -->

<script lang="ts">
	import { uniqueId } from 'underscore';
    import { application, blocks, state } from './stores';
    import { getContext, setContext } from 'svelte';
    import { writable, type Writable } from 'svelte/store';

    const { interface_id, add_block } = getContext('Interface');
    
    export let klass: string = "";
	export let base_datapanel_id: string;
	export let datapanel_id: string = base_datapanel_id;
	export let block_id: string = uniqueId('block-');
    export let block_element: HTMLElement | undefined = undefined;
    export let style: string = "";

    add_block(block_id, block_element, base_datapanel_id);
    setContext('Block', {block_id, interface_id, base_datapanel_id, block_element});

    $state[block_id] = { 
        datapanel_id: datapanel_id,
    };

    // // Add the block to the application store
    // $application.blocks.block_id = {
    //     'element': block_element,
    // };
 
    // $blocks.block_id = {
    //     'element': block_element,
    //     'interface_id': interface_id,
    // };
</script>

<div bind:this={block_element} class={klass} style={style}>
	<slot />
</div>

<script lang="ts">
	import { setContext } from 'svelte';
	import { uniqueId } from 'underscore';
	import { application } from './stores';

	export let interface_id: string = uniqueId('interface-');
	export let interface_element: HTMLElement | undefined = undefined;
    export let klass: string = "";
    export let style: string = "";
	let blocks: Array<BlockInterface> = [];

	interface BlockInterface {
		block_id: string;
		block_element: string;
		base_dataframe_id: string;
	}

	function add_block(block_id: string, block_element: string, base_dataframe_id: string): void {
		blocks = [...blocks, { block_id, block_element, base_dataframe_id }];
        $application.interfaces[interface_id].blocks.push(block_id);
	}

	setContext('Interface', { interface_id, add_block });

	// Add the interface to the application store
	$application.interfaces[interface_id] = {
		element: interface_element,
        blocks: []
	};
</script>


<div bind:this={interface_element} class={klass} style={style}>
	<slot />
</div>

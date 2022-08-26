<script lang="ts">
	import { uniqueId } from 'underscore';
    import { application } from './stores';

	export let base_datapanel_id: string;
	export let datapanel_id: string;
	export let group_id: string = uniqueId('block-');
    export let group_element: HTMLElement;

    // Add the block to the application store
    $application.block_groups.group_id = {
        'element': group_element,
    };
</script>

<div bind:this={group_element}>
	<slot />
</div>

<script lang="ts">
    import { writable } from "svelte/store";
    import Filter from "./Filter.svelte";
    import Gallery from "./Gallery.svelte";
    import NewBlock from "./NewBlock.svelte";
    import { _boxes } from "./stores.js";

    // Setup boxes
    // TODO this should be done by getting info from Python
    // console.log("here");
	_boxes['box_1'] = writable('box_1');

</script>
<!-- <NewBlock alias_to_box_id={new Map([['dp', 'box_1']])}>
    <Gallery dp="box_1"/>
</NewBlock>
<NewBlock alias_to_box_id={new Map([['dp', 'box_1']])}>
    <Filter/>
</NewBlock> -->

<NewBlock>
    <Gallery dp_box_id="box_1"/>
</NewBlock>

<NewBlock> 
    <Filter dp_box_id="box_1"/>
</NewBlock>
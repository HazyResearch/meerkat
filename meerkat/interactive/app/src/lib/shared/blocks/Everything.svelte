<script lang="ts">
    import { writable } from "svelte/store";
    import Filter from "./Filter.svelte";
    import Gallery from "./Gallery.svelte";
    import NewBlock from "./NewBlock.svelte";
    import { _refs } from "./stores.js";

    // Setup refs
    // TODO this should be done by getting info from Python
    // console.log("here");
	_refs['ref_1'] = writable('ref_1');

</script>
<!-- <NewBlock alias_to_ref_id={new Map([['df', 'ref_1']])}>
    <Gallery df="ref_1"/>
</NewBlock>
<NewBlock alias_to_ref_id={new Map([['df', 'ref_1']])}>
    <Filter/>
</NewBlock> -->

<NewBlock>
    <Gallery df_ref_id="ref_1"/>
</NewBlock>

<NewBlock> 
    <Filter df_ref_id="ref_1"/>
</NewBlock>
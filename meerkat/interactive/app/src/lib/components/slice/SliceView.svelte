
<script lang="ts">
    
    export let datapanel_id: string;
    export let main_column: string;
    export let tag_columns: Array<string>; 


</script>

<div class="isolate bg-white container mx-auto space-y-2 p-2">
    {#each gallery.slices as slice}
        <Panel name={slice.name} descriptions={slice.descriptions} stats={slice.stats} instances={slice.instances} plot={slice.plot}/>
    {/each}
</div>
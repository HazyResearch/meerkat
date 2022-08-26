<svelte:options accessors={true}/>
<script lang="ts">

    export let against: any;

</script>


<div class="bg-slate-100">
    <input type="text" bind:value={$against}>
    Match {$against}
</div>
<script lang="ts">


    let dragger;

    const reset = handler((start, p) => {
        min = clamp(Math.min(start.p, p), 0, 1);
        max = clamp(Math.max(start.p, p), 0, 1);
    });

</script>

<div bind:this={dragger} class="drag-select-outer" on:mousedown|stopPropagation={reset}>
    
</div>

<style>
    .drag-select-outer {

    }
</style>
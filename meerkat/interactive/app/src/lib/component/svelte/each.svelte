<script>
    export let items;
</script>

{#each items as item}
    <slot {item} />
{/each}
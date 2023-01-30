<script lang="ts">
    import { getContext } from 'svelte';

    export let label: string;
    export let id: string; 

    const { activeTabId, addTab } = getContext('tabs')
    addTab(label=label, id=id)
</script>
{#if $activeTabId === id}
<slot></slot>
{/if}

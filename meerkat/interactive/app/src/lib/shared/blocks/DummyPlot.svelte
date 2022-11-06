<script lang="ts">
    import { getContext } from "svelte";

    const { get_df } = getContext('Block');
    export let count;

</script>

<div class="text-violet-400">
    DummyPlot
</div>
<div class="text-violet-400 mb-8">
    <div>df: {$get_df('df', `q${count % 2 + 1}`)}</div>
</div>

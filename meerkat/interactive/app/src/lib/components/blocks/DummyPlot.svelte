<script lang="ts">
    import { getContext } from "svelte";

    const { get_dp } = getContext('Block');
    export let count;

</script>

<div class="text-violet-400">
    DummyPlot
</div>
<div class="text-violet-400 mb-8">
    <div>dp: {$get_dp('dp', `q${count % 2 + 1}`)}</div>
</div>

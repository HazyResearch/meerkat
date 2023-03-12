<script lang="ts">
	import type { Endpoint } from "$lib/utils/types";


    export let data: string;
    export let classes: string = '';

    export let on_scroll: Endpoint = null;
    export let on_scroll_axis_change: Endpoint = null;
</script>


<div>
    <img class={classes} src={data} alt="A Meerkat cell." />
</div>

<script>
    import Meerkat from "@meerkat-ml/meerkat/Meerkat.svelte";
    import Modals from '$lib/shared/common/Modals.svelte';
	import ComponentContext from '$lib/ComponentContext.svelte';
    import "$lib/app.css";
</script>

<ComponentContext>
    <Meerkat>
        <slot />
        <Modals />
    </Meerkat>
</ComponentContext>

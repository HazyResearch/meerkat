<script>
    import "$lib/app.css";
    import ComponentContext from '$lib/ComponentContext.svelte';
    import { Modals } from 'svelte-modals';
</script>

<ComponentContext>
    <slot />
    <Modals>
        <div slot="backdrop" class="fixed z-50 top-0 bottom-0 left-0 right-0 bg-black opacity-50" />
    </Modals>
</ComponentContext>

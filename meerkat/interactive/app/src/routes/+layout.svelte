<script lang="ts">
	import '$lib/app.css';
	import ComponentContext from '$lib/ComponentContext.svelte';
	import Modals from '$lib/shared/common/Modals.svelte';
</script>

<ComponentContext>
	<slot />
	<Modals />
</ComponentContext>

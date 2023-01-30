<script lang="ts">
	import ComponentContext from '$lib/ComponentContext.svelte';
	import Meerkat from '$lib/Meerkat.svelte';
	import Modals from '$lib/shared/common/Modals.svelte';

	import '$lib/app.css';
</script>

<ComponentContext>
	<Meerkat>
		<slot />
		<Modals />
	</Meerkat>
</ComponentContext>

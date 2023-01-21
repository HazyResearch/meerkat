<script lang="ts">
	import Meerkat from '$lib/Meerkat.svelte';
	import Modals from '$lib/shared/common/Modals.svelte';
	import '$lib/app.css';
</script>

<Meerkat>
	<slot />
	<Modals />
</Meerkat>

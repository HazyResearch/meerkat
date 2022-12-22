<script lang="ts">
	import Meerkat from '$lib/Meerkat.svelte';
	import Modals from '$lib/shared/common/Modals.svelte';
	import '../app.css';
</script>

<Meerkat>
	<slot />
</Meerkat>
<Modals />

<script lang="ts">
	import { onMount } from 'svelte';
	import Prism from 'prismjs';

	export let data: string;
</script>

<img
	class="block w-full rounded-md shadow-lg"
	src={data}
	alt="A cell."
/>

<script lang="ts">
	import { onMount } from 'svelte';
	import Prism from 'prismjs';

	export let data: string;
</script>

<img
	class="block object-center h-4/5 w-full rounded-md shadow-lg"
	src={data}
	alt="A cell."
/>

<script lang="ts">
	import '$lib/app.css';
	import ComponentContext from '$lib/ComponentContext.svelte';
	import Modals from '$lib/shared/common/Modals.svelte';
	import MediaPLayer from '$lib/shared/MediaPlayer.svelte';

</script>

<svelte:head>
    <script src="https://cdn.tailwindcss.com"></script>
</svelte:head>

<ComponentContext>
	<slot />
	<Modals />
</ComponentContext>

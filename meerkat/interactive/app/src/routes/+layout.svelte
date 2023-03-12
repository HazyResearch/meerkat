<script lang="ts">
	import '$lib/app.css';
	import ComponentContext from '$lib/ComponentContext.svelte';
	import Modals from '$lib/shared/common/Modals.svelte';
	import MediaPlayer from '$lib/shared/media/MediaPlayer.svelte';


</script>

<svelte:head>
    <script src="https://cdn.tailwindcss.com"></script>
</svelte:head>

<MediaPlayer/>
<ComponentContext>
	<slot />
	<Modals />
</ComponentContext>

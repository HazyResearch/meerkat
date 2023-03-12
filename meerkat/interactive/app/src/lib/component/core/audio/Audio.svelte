<script lang="ts">
	import { PlayFill, Pause } from 'svelte-bootstrap-icons';
	import { setCurrentMedia, currentMedia } from '$lib/shared/media/current';

	export let data: string;
	export let classes: string = '';

	const startPlay = () => {
		setCurrentMedia(data);
		if ($currentMedia) {
			$currentMedia.paused = false;
			$currentMedia.currentTime = 0;
		}
	};

	// FIXME: This should be based of the primary key of the cell.
	$: playing = $currentMedia && $currentMedia.data === data;
</script>

<div
	class={'h-full w-10 bg-slate-100 rounded-sm flex items-center justify-center self-center border' +
		classes}
	class:border-violet-600={playing}
>
	{#if playing}
		{#if $currentMedia.paused}
			<button on:click={() => ($currentMedia.paused = false)}>
				<PlayFill class="text-slate-600" />
			</button>
		{:else if $currentMedia.ended}
			<button on:click={startPlay}> <PlayFill class="text-slate-600" /> </button>
		{:else}
			<button on:click={() => ($currentMedia.paused = true)}>
				<Pause class="text-slate-600" />
			</button>
		{/if}
	{:else}
		<button on:click={startPlay}> <PlayFill class="text-slate-600" /> </button>
	{/if}
</div>

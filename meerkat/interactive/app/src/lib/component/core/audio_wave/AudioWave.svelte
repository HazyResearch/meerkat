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

	// FIXME(Sabri): This should be based of the primary key of the cell.
	$: playing = $currentMedia && $currentMedia.data === data;
</script>


TODO: Waveform

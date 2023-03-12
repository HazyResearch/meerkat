<script lang="ts">
	import { Soundwave } from 'svelte-bootstrap-icons';
	import { currentMedia } from './current';

	let showAudio = false;
</script>

{#if $currentMedia}
	<div
		class="bg-slate-100 z-20 absolute right-5 top-5 shadow-lg rounded-md px-1 border flex items-center"
		on:mouseover={() => (showAudio = true)}
		on:mouseleave={() => (showAudio = false)}
        on:focus={() => (showAudio = true)}
	>
		<Soundwave width={32} height={32} class="text-violet-600"/>
		<audio 
            controls={true} autoplay={true} src={$currentMedia.data} 
            class:hidden={!showAudio}
            bind:currentTime={$currentMedia.currentTime}
            bind:duration={$currentMedia.duration}
            bind:paused={$currentMedia.paused}
            bind:ended={$currentMedia.ended}
        />
        
	</div>

{/if}

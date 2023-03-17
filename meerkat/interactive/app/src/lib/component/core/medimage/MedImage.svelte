<script lang="ts">
	import type { Endpoint } from '$lib/utils/types';
	import { Fullscreen, FullscreenExit, PlayFill, PauseFill } from 'svelte-bootstrap-icons';

	export let data: Array<string>;
	export let classes: string = '';
	export let numSlices: number = null;
	export let showToolbar: boolean = false;
	export let fps: number = 20;

	export let onViewChange: Endpoint = null;

	// Whether the image is fullscreen.
	let isFullscreen: boolean = false;
	// Whether the video is playing.
	let isPlaying: boolean = false;
	// Whether the toolbar information should be shown.
	let isToolbarActive: boolean = false;

    // Transform properties.
    let scale = 1.0;
    let x = 0;
    let y = 0;

	// TODO: this should be reactive.
	if (numSlices === null) {
		numSlices = data.length;
	}
	let sliceNumber: number = Math.floor(numSlices / 2);

	function boundSliceNumber(sliceNumber: number) {
		return Math.min(Math.max(0, sliceNumber), numSlices - 1);
	}

	function handleScroll(event: WheelEvent) {
		if (numSlices === 1) {
			return;
		}
		sliceNumber += event.deltaY * 0.1;
		sliceNumber = Math.round(sliceNumber);

		// Restrict slices to the range [0, numSlices-1].
		sliceNumber = boundSliceNumber(sliceNumber);
	}

	function handleKeyPressFullscreen(event: KeyboardEvent) {
		console.log(event);
		if (event.key === 'Escape') {
			isFullscreen = false;
		} else if (event.key === 'ArrowUp') {
			sliceNumber = boundSliceNumber(sliceNumber - 1);
		} else if (event.key === 'ArrowDown') {
			sliceNumber = boundSliceNumber(sliceNumber + 1);
		}
	}

	function toggleFullscreen() {
		isFullscreen = !isFullscreen;
		if (isFullscreen) {
			window.addEventListener('keydown', handleKeyPressFullscreen);
		} else {
			window.removeEventListener('keydown', handleKeyPressFullscreen);
		}
	}

	function play() {
		isPlaying = true;
		nextImage();
	}

	function nextImage() {
		if (isPlaying) {
			sliceNumber = (sliceNumber + 1) % numSlices;
			setTimeout(nextImage, 1000 / fps); // change @ 30 fps
		}
	}
</script>

{#if showToolbar}
	<div class={isFullscreen ? 'fullscreen' : ''}>
		<div class="image-container w-full h-full">
			<img
				src={data[sliceNumber]}
				class={classes}
				on:wheel|preventDefault={handleScroll}
				alt="A medical image."
                style="width: 100%; height: 100%; object-fit: contain;"
			/>

			<!-- Top toolbar -->
			<div
				class="toolbar top-0 w-full"
				on:mouseenter={() => (isToolbarActive = true)}
				on:mouseleave={() => (isToolbarActive = false)}
				on:wheel|preventDefault={handleScroll}
			>
				<!-- Slice label -->
				<div class="button-container items-center">
					{#if isToolbarActive}
						<span
							class="flex-1 px-1"
							style="color: white; font-size: 0.8rem; font-weight: bold;"
						>
							Slice {sliceNumber + 1}/{numSlices}
						</span>
					{/if}
				</div>
			</div>

			<!-- Bottom toolbar -->
			<!-- Play button -->
			<!-- TODO fix handling of space bar key press -->
			<div
				class="toolbar bottom-1 w-full"
				on:mouseenter={() => (isToolbarActive = true)}
				on:mouseleave={() => (isToolbarActive = false)}
				on:wheel|preventDefault={handleScroll}
			>
				<div class="button-container mx-1">
					{#if isToolbarActive}
						{#if isPlaying}
							<button
								class="flex-1"
								on:click={() => {
									isPlaying = false;
								}}
							>
								<PauseFill width={24} height={24} fill="white" />
							</button>
						{:else}
							<button class="flex-1" on:click={play}>
								<PlayFill width={24} height={24} fill="white" />
							</button>
						{/if}

						<!-- Fullscreen button -->
						<button on:click={toggleFullscreen}>
							{#if isFullscreen}
								<FullscreenExit width={24} height={24} fill="white" />
							{:else}
								<Fullscreen width={24} height={24} fill="white" />
							{/if}
						</button>
					{/if}
				</div>
			</div>
		</div>
	</div>
{:else}
	<img
		on:wheel|preventDefault={handleScroll}
		class={classes}
		src={data[sliceNumber]}
		alt="A medical image."
	/>
{/if}

<style>
	.fullscreen {
		position: fixed;
		top: 0;
		left: 0;
		z-index: 2;
		width: 100%;
		height: 100%;
		background: black;
		display: flex;
		justify-content: center;
		align-items: center;
	}

	/* .fullscreen img {
		max-width: 100%;
		max-height: 100%;
	} */

	.image-container {
		position: relative;
	}

	.button-container {
		position: relative;
		z-index: 3;
		display: flex;
		/* justify-content: flex-end; */
		width: 100%;
		height: 100%;
	}

	.toolbar {
		position: absolute;
		z-index: 3;
		display: flex;
		justify-content: flex-end;
		padding: 0.1rem;
		height: 10%;
	}

	.overlay {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		/* background-color: rgba(76, 74, 74, 0.5); */
		display: flex;
		justify-content: center;
		align-items: center;
	}
</style>

<script lang="ts">
	import type { CellInfo } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import {
		Fullscreen,
		FullscreenExit,
		PlayFill,
		PauseFill,
		Compass,
		Palette
	} from 'svelte-bootstrap-icons';
	import { dispatch } from '$lib/utils/api';
	import { BarLoader } from 'svelte-loading-spinners';
	import Toolbar from '$lib/shared/common/Toolbar.svelte';

	export let data: Array<string>;
	export let classes: string = '';
	export let dim: number;
	export let showToolbar: boolean = false;
	export let fps: number = 20;
	export let onFetch: Endpoint;
	export let segmentationColumn: string = '';

	// Information about the cell
	export let cellInfo: CellInfo;

	// Cache the data for different dimensions.
	// TODO: check if this affects performance.
	const dataCache: any = {};
	dataCache[dim] = data;

	// Segmentation data (that can optionally be loaded in)
	let isSegmentationActive: boolean = false;
	let segmentation: Array<string> = null;
	const segmentationCache: any = {};

	// Whether the image is fullscreen.
	let isFullscreen: boolean = false;
	// Whether the video is playing.
	let isPlaying: boolean = false;
	// Whether the toolbar information should be shown.
	let isToolbarActive: boolean = false;
	// Whether we should pin the toolbar.
	let pinToolbar: boolean = false;
	// Status of loading operations.
	let status: string = '';

	// TODO: this should be reactive.
	$: numSlices = data.length;
	$: sliceNumber = Math.floor(numSlices / 2);

	function fetchData() {
		if (dim in dataCache) {
			data = dataCache[dim];
			return;
		}
		status = 'working';
		let promise = dispatch(onFetch.endpointId, {
			detail: { df: cellInfo.dfRefId, column: cellInfo.columnName, index: cellInfo.row, dim: dim }
		});
		promise
			.then((result) => {
				status = 'success';
				dataCache[dim] = result;
				data = result;
			})
			.catch(async (error) => {
				status = 'error';
			});
		fetchSegmentation();
	}

	function fetchSegmentation() {
		if (!isSegmentationActive) {
			return;
		}

		if (dim in segmentationCache) {
			segmentation = segmentationCache[dim];
			return;
		}
		status = 'working';
		let promise = dispatch(onFetch.endpointId, {
			detail: {
				df: cellInfo.dfRefId,
				column: segmentationColumn,
				index: cellInfo.row,
				dim: dim,
				type: 'segmentation'
			}
		});
		promise.then((result) => {
			status = 'success';
			segmentationCache[dim] = result;
			segmentation = result;
		});
	}

	function clampSliceNumber(sliceNumber: number) {
		return Math.min(Math.max(0, sliceNumber), numSlices - 1);
	}

	function handleScroll(event: WheelEvent) {
		if (numSlices === 1) {
			return;
		}
		sliceNumber += event.deltaY * 0.15;
		sliceNumber = Math.round(sliceNumber);

		// Restrict slices to the range [0, numSlices-1].
		sliceNumber = clampSliceNumber(sliceNumber);
	}

	function handleKeyPressFullscreen(event: KeyboardEvent) {
		console.log(event);
		if (event.key === 'Escape') {
			isFullscreen = false;
		} else if (event.key === 'ArrowUp') {
			sliceNumber = clampSliceNumber(sliceNumber - 1);
		} else if (event.key === 'ArrowDown') {
			sliceNumber = clampSliceNumber(sliceNumber + 1);
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

	/**
	 * Play the frames as a video.
	 */
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
	<div
		class={isFullscreen
			? 'fullscreen'
			: 'w-full h-full ' + (status === 'working' ? '' : 'bg-black')}
	>
		{#if status === 'working'}
			<div class="flex justify-center items-center h-full">
				<BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
			</div>
		{:else if status === 'error'}
			<div class="flex justify-center items-center h-full">Error</div>
		{:else}
			<div class="image-container w-full h-full">
				<div class="w-full h-full" style="position: relative;">
					<img
						src={data[sliceNumber]}
						class={classes + 'z-index-0'}
						on:wheel|preventDefault={handleScroll}
						alt="A medical image."
						style="position:absolute; : 0; left: 0; width: 100%; height: 100%; object-fit: contain;"
					/>
					{#if isSegmentationActive}
						<img
							src={segmentation[sliceNumber]}
							class={classes + 'z-index-2'}
							on:wheel|preventDefault={handleScroll}
							alt="A segmentation."
							style="opacity: 0.7; position:absolute; : 0; left: 0; width: 100%; height: 100%; object-fit: contain;"
						/>
					{/if}
				</div>

				<!-- Top toolbar -->
				<Toolbar on:wheel={handleScroll} bind:isToolbarActive pin={pinToolbar} classes="px-3">
					<span class="" style="color: white; font-size: 0.8rem; font-weight: bold;">
						Slice {sliceNumber + 1}/{numSlices}
					</span>

					<div class={'grid gap-x-3 grid-cols-' +  + (1 + (segmentationColumn !== ""))}>
						<!-- Reformat button -->
						<button
							on:click={() => {
								dim = (dim + 1) % 3;
								fetchData();
							}}
						>
							<Compass width={24} height={24} fill="white" />
						</button>

						<!-- Segmentation button -->
						{#if segmentationColumn != ''}
							<button
								on:click={() => {
									isSegmentationActive = !isSegmentationActive;
									fetchSegmentation();
								}}
							>
								<Palette width={24} height={24} fill={isSegmentationActive ? 'yellow' : 'white'}/>
							</button>
						{/if}
					</div>
				</Toolbar>

				<!-- Bottom toolbar -->
				<!-- Play button -->
				<!-- TODO fix handling of space bar key press -->
				<Toolbar
					on:wheel={handleScroll}
					bind:isToolbarActive
					pin={pinToolbar}
					classes="px-3"
					align="bottom"
				>
					{#if isPlaying}
						<button
							class=""
							on:click={() => {
								isPlaying = false;
							}}
						>
							<PauseFill width={24} height={24} fill="white" />
						</button>
					{:else}
						<button class="" on:click={play}>
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
				</Toolbar>
			</div>
		{/if}
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
		justify-content: flex-end;
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

<script lang="ts">
	import { API_URL } from '$lib/constants';
	import { Progressbar } from 'flowbite-svelte';
	import { get } from 'svelte/store';

	export let progress: number = 0;
	export let info: string = 'Server Ready.';
	export let running: boolean = false;

	// Fetch data from this streaming endpoint
	const eventSource = new EventSource(`${get(API_URL)}/subscribe/progress/`);

	eventSource.addEventListener('endpoint', async (event) => {
		let endpoint = JSON.parse(event.data);
		running = true;
		progress = 0;
		info = `Running endpoint ${endpoint}...`;
		console.log('Endpoint', endpoint);
	});

	let startPayload;
	eventSource.addEventListener('start', (event) => {
		startPayload = JSON.parse(event.data);
		running = true;
		console.log('Start', startPayload);
	});

	eventSource.addEventListener('progress', (event) => {
		let op;
		({ op, progress } = JSON.parse(event.data));
		info = op !== 'Done!' ? `Running ${op}...` : op;
		console.log('Progress', op, progress);
	});

	eventSource.addEventListener('end', async (event) => {
		console.log('End');
		await new Promise((r) => setTimeout(r, 500));
		running = false;
		// Removing this `progress=0` causes events
		// to not be triggered correctly!
		// FIXME: debug this
		// progress = 0;
		info = 'Server Ready.';
	});
</script>

<div class="h-12 mb-1">
	{#if running}
		<div
			class="mb-2 pt-1 bg-purple-50 rounded-lg px-2 text-sm font-medium text-purple-700"
		>
			{info}
			<div class="flex items-center justify-between mb-1">
				<Progressbar progress={progress.toString()} color="purple" size="h-2" class="mr-4" />
				<span class="text-sm">{progress}%</span>
			</div>
		</div>
	{:else}
		<div
			class="text-center mb-2 py-1 bg-green-50 rounded-lg px-2 text-sm font-medium text-green-700"
		>
			{info}
		</div>
	{/if}
</div>

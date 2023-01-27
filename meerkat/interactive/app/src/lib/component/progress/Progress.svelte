<script lang="ts">
	import { API_URL } from '$lib/constants';
	import { Progressbar } from 'flowbite-svelte';
	import { get } from 'svelte/store';

	export let progress: number = 0;
	export let info: string = 'Server Ready.';
	export let running: boolean = false;

    // Fetch data from this streaming endpoint
    const eventSource = new EventSource(`${get(API_URL)}/subscribe/progress/`);

    let start_payload;
    eventSource.addEventListener("start", (event) => {
        start_payload = JSON.parse(event.data);
        running = true;
        console.log("Start", start_payload)
    });

    eventSource.addEventListener("progress", (event) => {
        let op;
        ({op, progress} = JSON.parse(event.data));
        console.log(op, progress);
        info = `Running ${op}...`;
    });

    eventSource.addEventListener("end", async (event) => {
        info = "Done!";
        progress = 100;
        await new Promise(r => setTimeout(r, 500));
        running = false;
        info = "Server Ready.";
    });

</script>

<div class="h-12">
    {#if running}
        <div class="mb-2 bg-purple-50 rounded-lg px-2 text-sm font-medium text-purple-700 dark:text-white">
            <span class="">{info}</span>
            <div class="flex items-center justify-between mb-1">
                <Progressbar progress={progress.toString()} color="purple" size="h-2" class="mr-4" />
                <span class="text-sm">{progress}%</span>
            </div>
        </div>
    {:else}
        <div class="mb-2 bg-green-50 rounded-lg px-2 text-sm font-medium text-green-700 dark:text-white">
            <span class="">{info}</span>
        </div>
    {/if}
</div>
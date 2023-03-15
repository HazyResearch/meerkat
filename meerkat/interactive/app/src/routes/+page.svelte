<script lang="ts">
	import banner from '$lib/assets/banner_small.png';
	import { API_URL } from '$lib/constants';
	import Page from '$lib/shared/Page.svelte';
	import type { PageType } from '$lib/utils/types';
	import { createWebSocket, WEBSOCKET } from '$lib/websocket';
	import { onMount } from 'svelte';

	let config: PageType | null = null;
	onMount(async () => {
		const id = new URLSearchParams(window.location.search).get('id');

		// Create a websocket connection to the server
		// Get a unique ID for this connection
		const me = Math.random().toString(36).substring(7);

		console.log(me);
		// Get the $API_URL and replace the http with ws
		const ws = await createWebSocket(`${$API_URL.replace('http', 'ws')}/ws/${me}/`);
		// const ws = new WebSocket(`${$API_URL.replace('http', 'ws')}/ws/${me}/`);

		// Wait for the connection.

		ws.send('Hello from the client!');
		// ws.onopen = () => {
		// 	console.log('connected');
		// 	ws.send('Hello from the client!');

		// };

		// Get a message from the server
		ws.onmessage = (msg) => {
			if (msg.data === 'no') {
				console.log('no');
			} else {
				console.log('YES');
				// Set the websocket connection.
				if (!$WEBSOCKET) {
					$WEBSOCKET = ws;
				}
			}
		};

		if ($WEBSOCKET) {
			ws.send(
				JSON.stringify({
					request: {
						method: 'GET',
						url: `${$API_URL}/page/${id}/config/`
					}
				})
			);
		} else {
			config = await (await fetch(`${$API_URL}/page/${id}/config/`)).json();
		}
		document.title = config?.name ? config.name : 'Meerkat';
	});
</script>

<div class="h-screen">
	{#if config}
		<Page {config} />
	{:else}
		<div class="flex justify-center h-screen items-center">
			<img src={banner} alt="Meerkat" class="h-12" />
		</div>
	{/if}
</div>

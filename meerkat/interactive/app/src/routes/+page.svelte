<script lang="ts">
	import banner from '$lib/assets/banner_small.png';
	import { API_URL } from '$lib/constants';
	import Interface from '$lib/shared/Interface.svelte';
	import type { InterfaceType } from '$lib/utils/types';
	import { onMount } from 'svelte';

	let config: InterfaceType | null = null;
	onMount(async () => {
		const id = new URLSearchParams(window.location.search).get('id');
		config = await (await fetch(`${$API_URL}/interface/${id}/config`)).json();
		document.title = config?.name ? config.name : 'Meerkat';
	});
</script>

<div class="h-screen p-3">
	{#if config}
		<Interface {config} />
	{:else}
		<div class="flex justify-center h-screen items-center">
			<img src={banner} alt="Meerkat" class="h-12" />
		</div>
	{/if}
</div>

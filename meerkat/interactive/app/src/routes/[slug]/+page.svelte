<script lang="ts">
	import banner from '$lib/assets/banner_small.png';
	import Interface from '$lib/shared/Interface.svelte';
	import type { InterfaceType } from '$lib/utils/types';
	import { onMount } from 'svelte';
	import { API_URL } from '$lib/constants';
	import { page } from '$app/stores';

	let config: InterfaceType | null = null;
	onMount(async () => {
		config = await (await fetch(`${$API_URL}/interface/${$page.params.slug}/config`)).json();
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

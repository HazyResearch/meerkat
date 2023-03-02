<script lang="ts">
	import { page } from '$app/stores';
	import banner from '$lib/assets/banner_small.png';
	import { API_URL } from '$lib/constants';
	import Page from '$lib/shared/Page.svelte';
	import type { PageType } from '$lib/utils/types';
	import { onMount } from 'svelte';

	let config: PageType | null = null;
	onMount(async () => {
		config = await (await fetch(`${$API_URL}/page/${$page.params.slug}/config/`)).json();
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

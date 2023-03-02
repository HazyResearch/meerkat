<script lang="ts">
	import banner from '$lib/assets/banner_small.png';
	import { API_URL } from '$lib/constants';
	import { Page } from '@meerkat-ml/meerkat';
	import { onMount } from 'svelte';

	let config: Page | null = null;
	onMount(async () => {
		const id = new URLSearchParams(window.location.search).get('id');
		config = await (await fetch(`${$API_URL}/page/${id}/config`)).json();
		document.title = config?.name ? config.name : 'Meerkat';
	});
</script>

<div class="h-screen">
	{% raw %}{#if config}{% endraw %}
		<Page {config} />
	{:else}
		<div class="flex justify-center h-screen items-center">
			<img src={banner} alt="Meerkat" class="h-12" />
		</div>
	{/if}
</div>

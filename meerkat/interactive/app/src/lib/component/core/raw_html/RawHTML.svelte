<script lang="ts">
	import sanitizeHtml from 'sanitize-html';
	import Globe2 from 'svelte-bootstrap-icons/lib/Globe2.svelte';

	export let html: string;
	export let view: string = 'full';
	console.log(html)

	$: sanitizedHtml = sanitizeHtml(html);
</script>

{#if view === 'icon'}
	<span><Globe2/></span>
{:else if view === 'thumbnail'}
	<div class="thumbnail h-full aspect-square rounded-md shadow-md border-black">
		<iframe
			srcdoc={sanitizedHtml}
			title={'title'}
			class="rounded-md"
			frameborder="0"
			style="height: 100%; width: 100%;"
		/>
	</div>
{:else}
	<div class="h-full w-full rounded-md shadow-md border-black">
		<iframe
			srcdoc={sanitizedHtml}
			title={'title'}
			class="rounded-md"
			frameborder="0"
			style="height: 100%; width: 100%;"
		/>
	</div>
{/if}

<style>
	.thumbnail:after {
		content: '';
		display: block;
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
	}
</style>

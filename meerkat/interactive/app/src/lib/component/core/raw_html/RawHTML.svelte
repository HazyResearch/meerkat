<script lang="ts">
	import sanitizeHtml from 'sanitize-html';
	import Globe2 from 'svelte-bootstrap-icons/lib/Globe2.svelte';

	export let html: string;
	export let view: string = 'full';
	export let sanitize: boolean = true;
	export let classes: string = "rounded-md shadow-md";

	let process = (html: string) => {
		if (sanitize) {
			return sanitizeHtml(html);
		} else {
			return html;
		}
	};
	$: sanitizedHtml = process(html);
</script>


{#if view === 'thumbnail'}
	<div class={"thumbnail h-full aspect-square " + classes}>
		<iframe
			srcdoc={sanitizedHtml}
			title={'title'}
			class="rounded-md"
			frameborder="0"
			style="height: 100%; width: 100%;"
		/>
	</div>
{:else}
	<div class={"h-full w-full " + classes}>
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

<script lang="ts">
	import Prism from 'prismjs';

	// TODO: Add all themes here
	type PrismTheme =
		| 'coy'
		| 'dark'
		| 'funky'
		| 'okaidia'
		| 'solarizedlight'
		| 'tomorrow'
		| 'twilight';

	export let body: string;
	export let language: string = 'python';
	export let theme: PrismTheme = 'okaidia';
	export let classes: string = '';
	export let background: string = "bg-slate-800";

	$: code = Prism.highlight(body, Prism.languages.js, language);
</script>
<svelte:head>
	<link
		href="https://cdnjs.cloudflare.com/ajax/libs/prism/9000.0.1/themes/prism-{theme}.min.css"
		rel="stylesheet"
	/>
</svelte:head>

<div
	class={"text-left whitespace-pre overflow-auto py-2 px-4 h-full aspect-video " + background + " " + classes}
>
	<pre><code class="language-{language}">{@html code}</code></pre>
</div>

<!-- h-full max-h-[50vh] my-1 rounded-lg -->

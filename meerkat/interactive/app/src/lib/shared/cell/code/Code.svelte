<script lang="ts">
	import Prism from 'prismjs';
	import 'prismjs/components/prism-python.js';

	export let data: string;
	export let language: any = 'python';
	export let theme: string = 'okaidia';

	let html_code = Prism.highlight(data, Prism.languages.js, language);
</script>

<svelte:head>
	<link
		href="https://cdnjs.cloudflare.com/ajax/libs/prism/9000.0.1/themes/prism-{theme}.min.css"
		rel="stylesheet"
	/>
</svelte:head>

<div class="whitespace-pre overflow-auto h-full max-h-[50vh] my-1 rounded-lg py-2 px-4 bg-black">
	<pre><code class="language-{language}">{@html html_code}</code></pre>
</div>

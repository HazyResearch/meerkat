<script lang="ts">
	import Prism from 'prismjs';
	import 'prismjs/components/prism-python.js';
	
	export let data: string;
	export let language: any = 'python';

	let html = Prism.highlight(data, Prism.languages[language]);
</script>

<svelte:head>
	<link rel="stylesheet" href="../../../../node_modules/prismjs/themes/prism-okaidia.css" />
</svelte:head>

<div class="whitespace-pre overflow-auto h-full my-1 rounded-lg py-2 px-4 bg-slate-600">
	<pre><code class={language}>{@html html}</code></pre>
</div>

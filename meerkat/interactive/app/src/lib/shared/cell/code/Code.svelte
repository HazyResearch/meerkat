<script lang="ts">
	import Prism from 'prismjs';
	import 'prismjs/components/prism-python.js';
	
	//export let data: string;
	export let data: Writable<string>;

	export let language: any = 'python';

	let html_code = Prism.highlight($data, Prism.languages[language]);
</script>

<svelte:head>
	<link rel="stylesheet" href="../../../../node_modules/prismjs/themes/prism-okaidia.css" />
</svelte:head>

<div class="whitespace-pre overflow-auto h-full max-h-[50vh] my-1 rounded-lg py-2 px-4 bg-orange-300">
	<pre><code class={language}>{@html $data}</code></pre>
</div>

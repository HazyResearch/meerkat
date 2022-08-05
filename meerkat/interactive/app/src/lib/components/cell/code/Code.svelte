<script lang="ts">
	import { onMount } from 'svelte';
	import Prism from 'prismjs';

	export let data: string;
	export let language: any = "js";


	let html = Prism.highlight(data, Prism.languages[language]);
</script>

<svelte:head>
	<link rel="stylesheet" href="../../../../node_modules/prismjs/themes/prism-okaidia.css">    
</svelte:head>

<div class="whitespace-pre overflow-scroll h-32 my-1 font-mono rounded-lg px-4 bg-slate-800">
	<pre><code class="{language}">{@html html}</code></pre>
</div>

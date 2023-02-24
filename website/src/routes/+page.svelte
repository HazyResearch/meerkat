<script lang="ts">
	import Nav from '../lib/Nav.svelte';
	import Content from '../lib/Content.svelte';
	import Footer from '../lib/Footer.svelte';
</script>

<div class="flex flex-col h-screen">
	<header class="fixed w-full top-0 z-50">
		<Nav />
	</header>
	<main class="relative grow">
		<Content />
		<!-- <Content /><Content /> -->

	</main>
	<footer class="sticky bottom-0 z-50">
		<Footer />
	</footer>
</div>
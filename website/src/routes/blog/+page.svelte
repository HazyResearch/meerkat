<script lang="ts">
	import Nav from '$lib/Nav.svelte';
	import Content from '$lib/Content.svelte';
	import Footer from '$lib/Footer.svelte';
	import Blog from '$lib/Blog-03-02-23.svelte';
</script>

<div class="flex flex-col h-screen">
	<header class="fixed w-full top-0 z-50">
		<Nav page="blog" />
	</header>
	<main class="relative grow">
		<Blog />
	</main>
	<footer class="sticky bottom-0 z-50">
		<Footer />
	</footer>
</div>
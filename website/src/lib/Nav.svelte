<script lang="ts">
	import { base } from '$app/paths';
	import { Button, Navbar, NavBrand, NavHamburger, NavLi, NavUl } from 'flowbite-svelte';

	export let page: string = 'home';
</script>

<Navbar
	navClass="font-rubik px-2 sm:px-4 py-2.5 border-b border-b-2 bg-gradient-to-r from-violet-300 via-red-200 to-orange-100"
	let:hidden
	let:toggle
>
	<NavBrand href="{base}/">
		<img src="{base}/favicon.png" class="mr-3 h-10" alt="Meerkat Logo" />
		<span class="text-violet-700 self-center whitespace-nowrap text-2xl font-light dark:text-white">
			Meerkat
		</span>
	</NavBrand>
	<div class="flex md:order-2">
		<Button
			size="sm"
			btnClass="bg-gradient-to-r from-violet-400 via-red-400 to-orange-400 hover:from-violet-500 hover:via-red-500 hover:to-orange-500 text-white focus:ring-4 focus:outline-none focus:ring-violet-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center mr-3 md:mr-0 dark:bg-violet-600 dark:hover:bg-violet-700 dark:focus:ring-violet-800"
		>
			Get started
		</Button>
		<NavHamburger on:click={toggle} />
	</div>
	<NavUl {hidden}>
		<NavLi href="{base}/" active={page === 'home'} activeClass="text-white md:bg-transparent">
			<span class:text-purple-500={page === 'home'} class="hover:text-purple-800"> Home </span>
		</NavLi>
		<NavLi href="{base}/blog" active={page === 'blog'}>
			<span class:text-purple-500={page === 'blog'} class="text-purple-500 hover:text-purple-800">
				Blog
			</span>
		</NavLi>
		<NavLi href="{base}/docs/index.html" active={page === 'docs'}>
			<span class:text-purple-500={page === 'docs'} class="text-purple-500 hover:text-purple-800">
				Documentation
			</span>
		</NavLi>
	</NavUl>
</Navbar>

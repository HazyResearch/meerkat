<script lang="ts">
	import { base } from '$app/paths';
	import { Button, Navbar, NavBrand, NavHamburger, NavLi, NavUl } from 'flowbite-svelte';

	export let page: string = 'home';
</script>

<Navbar
	navClass="font-rubik px-2 sm:px-4 py-2.5 border-b !border-b-1 !bg-slate-50 drop-shadow-sm"
	navDivClass='mx-auto flex flex-wrap justify-between items-center'
	let:hidden
	let:toggle
>
	<NavBrand href="{base}/">
		<img src="{base}/favicon.png" class="mr-3 h-10" alt="Meerkat Logo" />
		<span class="text-violet-700 self-center whitespace-nowrap text-2xl font-light dark:text-white">
			Meerkat
		</span>
	</NavBrand>
	<div class="flex flex-wrap my-1 sm:my-0 md:order-2">
		<a href="https://github.com/hazyresearch/meerkat" target="_blank" class="ml-4 mr-8 my-1 sm:my-0 self-center">
			<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
		</a>
		<Button
			size="sm"
			href="{base}/docs/index.html"
			btnClass="bg-gradient-to-r from-violet-600 via-pink-500 to-orange-400 hover:from-violet-500 hover:via-red-500 hover:to-orange-500 text-white focus:ring-4 focus:outline-none focus:ring-violet-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center mr-3 md:mr-0 dark:bg-violet-600 dark:hover:bg-violet-700 dark:focus:ring-violet-800"
		>
			Get started
		</Button>
		<!-- Add a Github Logo that when clicked goes to the repo. -->
		<NavHamburger on:click={toggle} />
	</div>
	<NavUl {hidden}>
		<NavLi href="{base}/" active={page === 'home'} activeClass="text-white md:bg-transparent">
			<span class:text-violet-500={page === 'home'} class="hover:text-violet-800"> Home </span>
		</NavLi>
		<NavLi href="{base}/blog" active={page === 'blog'}>
			<span class:text-violet-500={page === 'blog'} class="text-violet-500 hover:text-violet-800">
				Blog
			</span>
		</NavLi>
		<NavLi href="{base}/docs/index.html" active={page === 'docs'}>
			<span class:text-violet-500={page === 'docs'} class="text-violet-500 hover:text-violet-800">
				Documentation
			</span>
		</NavLi>
	</NavUl>
</Navbar>

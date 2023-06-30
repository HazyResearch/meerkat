<script lang="ts">
	import { getContext, onMount, type ComponentType } from 'svelte';
	import { writable } from 'svelte/store';

	const components: { [key: string]: ComponentType } = getContext('Components');

	export let name: string;
	export let props: any;
	export let slots: any = [];


	let component: ComponentType;
	onMount(async () => {
		// If the library is Meerkat, then we can load the component
		// directly from the component map
		if (name in components) {
			component = components[name];
			return;
		}
	});
	$:{
		component = components[name];
	}

</script>


{#if component}
	<!-- Pass the props to the component being rendered -->
	<svelte:component this={component} {...props} on:edit>
		{#each slots as slot}
			<svelte:self {...slot}/>
		{/each}
	</svelte:component>
{/if}

<script lang="ts">
	import { LogicalPartition } from 'carbon-icons-svelte';
	import { getContext, onMount, type ComponentType } from 'svelte';

	const components: { [key: string]: ComponentType } = getContext('Components');

	export let component_id: string;
	export let name: string;
	export let path: string;
	export let props: any;
	export let slots: any;
	export let library: string;

	let component: ComponentType;
	onMount(async () => {
		// If the library is Meerkat, then we can load the component
		// directly from the component map
		if (name in components) {
			component = components[name];
			return;
		} else {
			console.log(`Component ${name} was not imported and added to the Components context.`);
		}
	});
</script>


{#if component}
	<!-- Pass the props to the component being rendered -->
	<svelte:component this={component} {...props}>
		{#each slots as slot}
			<svelte:self {...slot} />
		{/each}
	</svelte:component>
{/if}

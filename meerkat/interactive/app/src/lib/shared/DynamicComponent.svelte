<script lang="ts">
	export let component_id: string;
	export let name: string;
	export let path: string;
	export let props: any;
</script>

{#await import(path)}
	Loading {name} component.
{:then Component}
	<svelte:component this={Component.default} {...props} />
{:catch error}
	{error}
{/await}

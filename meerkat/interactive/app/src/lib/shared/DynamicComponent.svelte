<script lang="ts">
	export let name: string;
	export let props: any;
</script>

{#await import(`$lib/component/${name.toLowerCase()}/${name}.svelte`)}
	Loading {name} component.
{:then Component}
	<svelte:component this={Component.default} {...props} />
{:catch error}
	{error}
{/await}

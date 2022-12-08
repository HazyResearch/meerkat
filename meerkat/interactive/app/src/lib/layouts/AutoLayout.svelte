<script lang="ts">
	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';

	export let components: Array<any>;
	export let classes: string;
</script>

<div class="flex flex-col {classes}">
	{#each components as component}
		<DynamicComponent {...component} />
	{/each}
</div>

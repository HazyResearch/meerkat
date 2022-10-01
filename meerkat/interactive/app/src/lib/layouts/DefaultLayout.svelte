<script lang="ts">
	import DynamicComponent from "$lib/common/DynamicComponent.svelte";
	import type {Component} from "$lib/utils/types";

	export let components: Array<Component>;;
</script>

<div class="flex flex-col space-y-2 h-screen p-3">
	{#each components as { name, props }}
		<DynamicComponent name={name} props={props} />
	{/each}
</div>

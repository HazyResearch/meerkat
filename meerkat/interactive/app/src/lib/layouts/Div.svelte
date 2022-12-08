<script lang="ts">
	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';

	export let component: Array<any>;
	export let classes: string;
</script>

<div class="{classes}">
    <DynamicComponent {...component} />	
</div>

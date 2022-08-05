<script lang="ts">
	import Code from './code/Code.svelte';
	import Image from './image/Image.svelte';

	export let data: any;
	export let cell_component: string;
	export let cell_props: object;
</script>

{#if cell_component === 'image'}
	<Image data={data} />
{:else if cell_component === 'code'}
	<Code data={data} {...cell_props} />
{:else}
	{data}
{/if} 
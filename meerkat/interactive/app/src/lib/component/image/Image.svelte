<script lang="ts">
	export let data: string;
	export let classes: string = '';
	export let thumbnail: boolean = false;
</script>

{#if thumbnail}
	This is a thumbnail
{:else}
	This is not a thumbnail	
{/if}

<img class={"h-full w-full" + classes} src={data} alt="A Meerkat cell." />

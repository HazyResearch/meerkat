<script lang="ts">
	export let classes: string;
	export let style: string;
</script>

<h1 class={classes} {style}>
    <slot />
</h1>
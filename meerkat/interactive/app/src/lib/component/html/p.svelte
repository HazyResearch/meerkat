<script lang="ts">
	export let classes: string;
	export let style: string;
</script>

<p class={classes} {style}>
	<slot />
</p>

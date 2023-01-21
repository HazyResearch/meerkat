<script lang="ts">
	export let classes: string;
	export let style: string;
	export let href: string;
</script>

<a class={classes} {style} {href}>
	<slot />
</a>

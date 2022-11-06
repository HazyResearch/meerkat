<script lang="ts">
	export let cssvars: object;

	$: style = Object.entries(cssvars)
		.map(([key, value]) => `--${key}:${value}`)
		.join(';');
</script>

<div style="display: contents; {style}">
	<slot />
</div>

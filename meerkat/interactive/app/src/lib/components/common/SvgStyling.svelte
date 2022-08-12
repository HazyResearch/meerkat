<script lang="ts">
	export let cssvars: object;

	$: style = Object.entries(cssvars)
		.map(([key, value]) => `--${key}:${value}`)
		.join(';');
</script>

<g {style}>
	<slot />
</g>

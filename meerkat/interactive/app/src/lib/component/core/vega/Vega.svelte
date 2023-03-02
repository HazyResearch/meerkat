<script lang="ts">
	import type { VisualizationSpec } from 'svelte-vega';
	import { Vega } from 'svelte-vega';
	import type { EmbedOptions } from 'vega-embed';

	export let data: Record<string, unknown>;
	export let spec: VisualizationSpec;
	export let options: EmbedOptions = {};
</script>

<div class="text-center bg-violet-50 rounded-lg p-4">
	<h2 class="font-mono text-xl font-bold">Vega</h2>
	<Vega {data} {spec} {options} />
</div>

<script lang="ts">
	import { getContext } from 'svelte';

	const { padding } = getContext('LayerCake');

	export let label: string = 'axis label';
    export let offset = 0;
</script>

<!-- Outer div is a container that spans the whole plot, 
    For y-axis, it's shifted to the left. -->

<div 
    class="container" 
    style:transform={`translate(-${$padding.left + offset}px, 0)`}
>
	<!-- Inner div styles the label.
        For y-axis, it rotates and translates it about a modified origin and positions it center left. -->
	<div class="ylabel">		
        {label}
	</div>
</div>

<style>
	.container {
		@apply absolute w-full h-full;
	}

	.ylabel {
        @apply absolute;
        @apply text-center font-bold text-slate-500;
		@apply left-0 top-1/2;
		transform-origin: center left;
		transform: rotate(-90deg) translate(-50%, -100%);
	}
</style>

<script lang="ts">
	import { getContext } from 'svelte';

	const { padding } = getContext('LayerCake');

	export let label: string = 'axis label';
    export let axis: 'x' | 'y' = 'x';
	export let inputable: boolean = false;
</script>

<!-- Outer div is a container that spans the whole plot, 
    For x-axis, it's shifted to the bottom.
    For y-axis, it's shifted to the left. -->

<div 
    class="container" 
    style:transform={axis === 'y' ? `translate(-${$padding.left}px, 0)`: ""}
>
	<!-- Inner div styles the label.
        For y-axis, it rotates and translates it about a modified origin and positions it center left. -->
	<div class="label {axis === 'x' ? 'xlabel' : 'ylabel'} w-full">
		{#if inputable}
			<input bind:value={label} />
		{:else}
			<div class="grid grid-cols-[auto_1fr_auto]">
				<div class="">v1</div>
				<div class="text-center">{label}</div>
				<div class=" ">v2</div>
			</div>
		{/if}
	</div>
</div>

<style>
	.container {
		@apply absolute w-full h-full;
	}

    .label {
        @apply absolute;
        @apply text-center font-bold text-slate-500;
    }

    .xlabel {
        @apply top-0 left-1/2;
        transform: translate(-160px, -55px);
    }

	.ylabel {
		@apply left-0 top-1/2;
		transform-origin: center left;
		transform: rotate(-90deg) translate(-50%, -100%);
	}
</style>

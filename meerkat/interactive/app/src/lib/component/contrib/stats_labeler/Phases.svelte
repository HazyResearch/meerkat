<script lang="ts">
	import OneCircle from 'svelte-bootstrap-icons/lib/_1Circle.svelte';
	import TwoCircle from 'svelte-bootstrap-icons/lib/_2Circle.svelte';
	import ThreeCircle from 'svelte-bootstrap-icons/lib/_3Circle.svelte';
	import CheckCircle from 'svelte-bootstrap-icons/lib/CheckCircle.svelte';
	import ArrowCounterclockwise from 'svelte-bootstrap-icons/lib/ArrowCounterclockwise.svelte';
	import { createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher();

	export let active_phase: string = 'train';
	export let phases: Array<string> = ['unassigned', 'train', 'precision', 'recall'];
</script>

<flex class="flex space-x-1 px-2">
	{#each ['train', 'precision', 'recall'] as phase, i}
		<button
			class="
                flex 
                items-center 
                justify-center  
                space-x-1
                w-28
                h-8 
                rounded-lg 
            "
			class:future={phases.indexOf(phase) > phases.indexOf(active_phase)}
			class:next={phases.indexOf(phase) === phases.indexOf(active_phase) + 1}
			class:done={phases.indexOf(phase) < phases.indexOf(active_phase)}
			class:active={phase === active_phase}
			class:shadow-lg={phase === active_phase}
			disabled={phases.indexOf(phase) != phases.indexOf(active_phase) + 1}
			on:click={() => dispatch('phase_change', phase)}
		>
			{#if phases.indexOf(phase) < phases.indexOf(active_phase)}
				<CheckCircle class="" width={18} height={18} />
			{:else if i === 0}
				<OneCircle class="" width={18} height={18} />
			{:else if i === 1}
				<TwoCircle class="" width={18} height={18} />
			{:else if i === 2}
				<ThreeCircle class="" width={18} height={18} />
			{/if}
			<div class="font-bold">
				{phase.charAt(0).toUpperCase() + phase.slice(1)}
			</div>
		</button>
	{/each}
	<button
		class="
            flex 
            items-center 
            justify-center
            self-end  
            space-x-1
            w-10
            h-8 
            rounded-lg 
            bg-red-200
            hover:shadow-lg
        "
        on:click={() => dispatch('phase_change', 'unassigned')}
	>
		<ArrowCounterclockwise class="self-center" width={18} height={18} />
	</button>
</flex>

<style>
	.active {
		@apply bg-violet-200 text-violet-800;
		@apply shadow-lg;
	}
	.done {
		@apply bg-green-200 text-green-800;
	}
	.future {
		@apply bg-gray-200 text-gray-800;
	}
	.next {
		@apply hover:shadow-lg;
	}
</style>

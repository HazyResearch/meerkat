<script lang="ts">
	import type { ComponentType } from '$lib/utils/types';
	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';


	export let gallery: ComponentType;
	export let gallery_match: ComponentType;
	export let gallery_filter: ComponentType;
	export let gallery_sort: ComponentType;
	export let discover: ComponentType;
	export let plot: ComponentType;
	export let active_slice: ComponentType;
	export let slice_sort: ComponentType;
	export let slice_match: ComponentType;
	export let global_stats: ComponentType;
	
</script>

<div class="grid grid-cols-[1fr_2fr] p-5 h-screen gap-5 max-width-100%">
	<div class="grid grid-rows-[auto_auto_auto_auto_auto_1fr] h-screen gap-5">
		<div class="bg-violet-200 py-1 rounded-lg drop-shadow-md z-40 flex flex-col">
			<div class="font-bold text-xl text-slate-800 self-center justify-self-center">
				<span color="transparent" text-shadow="0 0 0 white">♨️</span> Mocha ChangeList
			</div>
		</div>
		<DynamicComponent {...slice_match} />
		<DynamicComponent {...slice_sort} />
		<DynamicComponent {...discover} />
		<DynamicComponent {...global_stats} />
		<DynamicComponent {...plot} />
	</div>
	<div class="grid grid-rows-[auto_auto_1fr] h-screen gap-5">
		<div class="bg-violet-200 py-1 rounded-lg drop-shadow-md z-40 flex flex-col">
			<div class="font-bold text-xl text-slate-800 self-center justify-self-center">
				Slice Focus View
			</div>
		</div>
		<div class="grid grid-rows-auto grid-flow-col gap-3">
			<div class="flex flex-col space-y-3">
				<DynamicComponent {...active_slice} />
				<!-- <StatsLabeler {...gallery_editor} /> -->
			</div>
			<div class="flex flex-col space-y-3">
				<DynamicComponent {...gallery_match} />

				<DynamicComponent {...gallery_filter} />
				<DynamicComponent {...gallery_sort} />
			</div>
		</div>
		<DynamicComponent {...gallery} />
	</div>
</div>



<script lang="ts">
	import type { ComponentType } from '$lib/utils/types';
	import DynamicComponent from '$lib/shared/DynamicComponent.svelte';
	import { InfoCircle } from 'svelte-bootstrap-icons';

	export let code_control: boolean = false;

	export let gallery: ComponentType;
	export let gallery_match: ComponentType;
	export let gallery_filter: ComponentType;
	export let gallery_fm_filter: ComponentType;
	export let gallery_sort: ComponentType;
	export let gallery_code: ComponentType;
	export let discover: ComponentType;
	export let plot: ComponentType;
	export let active_slice: ComponentType;
	export let slice_sort: ComponentType;
	export let slice_match: ComponentType;
	export let global_stats: ComponentType;
	
</script>

<div class="grid grid-cols-[1fr_2fr] p-5 h-screen gap-12 max-width-100%">
	<div class="grid grid-rows-[auto_auto_auto_auto_auto_1fr] h-screen gap-2">
		<div class=" py-1 rounded-lg  z-40 flex flex-col">
			<div class="font-bold font-mono text-xl text-slate-800 self-center justify-self-center">
				ChangeList
			</div>
		</div>
		<DynamicComponent {...global_stats} />
		<DynamicComponent {...slice_match} />
		<DynamicComponent {...slice_sort} />

		<!-- <DynamicComponent {...discover} /> -->
		<div class="px-3 py-1 bg-slate-100 rounded-md flex gap-4 items-center">
			<InfoCircle width={32} height={32} class="text-violet-800"/>
			<div class="text-slate-800 text-left text-sm">
				The plot below shows changes in performance across different data slices.
				Click on a slice of data to bring it into focus on the right.
			</div>
		</div>
		<DynamicComponent {...plot} />
	</div>
	<div class="grid grid-rows-[auto_auto_1fr] h-screen gap-5">
		<div class=" py-1 rounded-lg  z-40 flex flex-col">
			<div class="font-bold text-xl text-slate-800 self-center justify-self-center">
				Slice Focus View
			</div>
		</div>
		<div class="grid grid-cols-[1fr_1fr] gap-3">
			<div class="flex flex-col space-y-3">
				<DynamicComponent {...active_slice} />
				<!-- <StatsLabeler {...gallery_editor} /> -->
			</div>
			<div class="flex flex-col space-y-3">
				{#if code_control}
					<DynamicComponent {...gallery_code} />
					<DynamicComponent {...gallery_fm_filter} />
				{:else}
					<DynamicComponent {...gallery_match} />
					<DynamicComponent {...gallery_filter} />
					<DynamicComponent {...gallery_sort} />
				{/if}
			</div>
		</div>
		<DynamicComponent {...gallery} />
	</div>
</div>



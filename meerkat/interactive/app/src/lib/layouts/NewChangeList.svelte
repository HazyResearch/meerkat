<script>
	import Plot from '$lib/component/plot/Plot.svelte';
	import Gallery from '$lib/component/gallery/Gallery.svelte';
	import Stats from '$lib/component/stats/Stats.svelte';
	import Match from '$lib/component/match/Match.svelte';
	import Filter from '$lib/component/filter/Filter.svelte';
	import Editor from '$lib/component/editor/Editor.svelte';
	import StatsLabeler from '$lib/component/stats_labeler/StatsLabeler.svelte';
	import Row from '$lib/component/row/Row.svelte';
	import Sort from '$lib/component/sort/Sort.svelte';

	export let components;

</script>

<div class="grid grid-cols-[1fr_2fr] p-5 h-screen gap-5 max-width-100%">
	<div class="grid grid-rows-[auto_auto_auto_auto_1fr] h-screen gap-5">
		<div class="bg-violet-200 py-1 rounded-lg drop-shadow-md z-40 flex flex-col">
			<div class="font-bold text-xl text-slate-800 self-center justify-self-center">
				<span color="transparent" text-shadow="0 0 0 white">☕️</span> Mocha ChangeList
			</div>
		</div>
		<Match {...components.slice_match.props} />
		<Sort {...components.slice_sort.props} />
		<Stats {...components.global_stats.props} />
		<Plot {...components.plot.props} />
	</div>
	<div class="grid grid-rows-[auto_auto_1fr] h-screen gap-5">
		<div class="bg-violet-200 py-1 rounded-lg drop-shadow-md z-40 flex flex-col">
			<div class="font-bold text-xl text-slate-800 self-center justify-self-center">
				Slice Focus View
			</div>
		</div>
		<div class="grid grid-rows-auto grid-flow-col gap-3">
			<div class="flex flex-col space-y-3">
				<Row {...components.active_slice.props} />
				<StatsLabeler {...components.gallery_editor.props} />
			</div>
			<div class="flex flex-col space-y-3">
				<Match {...gallery_match.props} />

				<Filter {...components.gallery_filter.props} />
				<Sort {...components.gallery_sort.props} />
			</div>
		</div>
		<Gallery {...components.gallery.props} />
	</div>
</div>

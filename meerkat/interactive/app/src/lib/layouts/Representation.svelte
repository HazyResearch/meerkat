<script>
	import Plot from '$lib/component/plot/Plot.svelte';
	import Gallery from '$lib/component/gallery/Gallery.svelte';
	import Stats from '$lib/component/stats/Stats.svelte';
	import Match from '$lib/component/match/Match.svelte';
	import Filter from '$lib/component/filter/Filter.svelte';
	import StatsLabeler from '$lib/component/stats_labeler/StatsLabeler.svelte';
	import Row from '$lib/component/row/Row.svelte';
	import Sort from '$lib/component/sort/Sort.svelte';
	import Editor from '$lib/component/editor/Editor.svelte';

	export let components;

	let gallery = components.gallery;
	let filter = components.filter;
	let gallery_match = components.match;
	let editor = components.editor;
	let plot = components.plot;
</script>

<div class="h-screen grid grid-rows-[auto_1fr] gap-4 py-6">
	<div class="grid grid-cols-[1fr_1fr] gap-x-3">
		<Plot {...plot.props} />
		<div class="grid grid-rows-[auto_auto_auto] gap-4">
			<Match {...gallery_match.props} />
			<Filter {...filter.props} />
			<Editor {...editor.props} />
		</div>
	</div>
	<Gallery {...gallery.props} />
</div>

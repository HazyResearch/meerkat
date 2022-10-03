<script>
	import Plot from '$lib/component/plot/Plot.svelte';
	import Gallery from '$lib/component/gallery/Gallery.svelte';
	import Stats from '$lib/component/stats/Stats.svelte';
	import Match from '$lib/component/match/Match.svelte';
	import Filter from '$lib/component/filter/Filter.svelte';
	import StatsLabeler from '$lib/component/stats_labeler/StatsLabeler.svelte';
	import Row from '$lib/component/row/Row.svelte';
	import Sort from '$lib/component/sort/Sort.svelte';

	export let components;

	let gallery = components.gallery;
	let gallery_match = components.match;
	let gallery_editor = components.gallery_editor;
	let plot = components.plot;
</script>

<div class="h-screen grid grid-rows-[auto_1fr] gap-4" >
    <Match {...gallery_match.props} />
    <Gallery {...gallery.props} />
</div>

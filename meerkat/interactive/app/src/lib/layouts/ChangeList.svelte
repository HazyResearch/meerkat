<script>
	import Plot from '$lib/component/plot/Plot.svelte';
	import Gallery from '$lib/component/gallery/Gallery.svelte';
	import Stats from '$lib/component/stats/Stats.svelte';
	import Match from '$lib/component/match/Match.svelte';
	import Filter from '$lib/component/filter/Filter.svelte';
	import Editor from '$lib/component/editor/Editor.svelte';

	export let components;

	let gallery_component = components.gallery;
	let gallery_match = components.gallery_match;
	let gallery_filter = components.gallery_filter;
	let gallery_editor = components.gallery_editor;
</script>

<div class="grid grid-rows-[1fr_2fr] py-10 h-screen gap-5">
	<div>
		<Match {...gallery_match.props} />
		<Filter {...gallery_filter.props} />
		<Editor {...gallery_editor.props} />
	</div>
	<Gallery {...gallery_component.props} />
</div>

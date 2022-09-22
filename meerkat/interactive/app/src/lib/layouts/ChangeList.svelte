<script>
	import Plot from '$lib/component/plot/Plot.svelte';
	import Gallery from '$lib/component/gallery/Gallery.svelte';
	import Stats from '$lib/component/stats/Stats.svelte';
	import Match from '$lib/component/match/Match.svelte';
	import Filter from '$lib/component/filter/Filter.svelte';
	import Editor from '$lib/component/editor/Editor.svelte';
	import Row from '$lib/component/row/Row.svelte';
	import Sort from '$lib/component/sort/Sort.svelte';

	export let components;

	let gallery = components.gallery;
	let gallery_match = components.gallery_match;
	let gallery_filter = components.gallery_filter;
	let gallery_sort = components.gallery_sort;
	let gallery_editor = components.gallery_editor;
	let plot = components.plot;
	let active_slice = components.active_slice;
	let slice_sort = components.slice_sort;
	let slice_match = components.slice_match;
</script>
<div class="grid grid-cols-[33%_1fr] p-5 h-screen gap-5">
	<div class="grid grid-rows-[auto_1fr] h-screen gap-5">
		<Match {...slice_match.props} />
		<Sort {...slice_sort.props} />
		<Plot {...plot.props} />
	</div>
	<div class="grid grid-rows-[auto_1fr] h-screen gap-5">
		<div class="grid grid-rows-1 grid-flow-col gap-3">
			<div class="flex flex-col space-y-3">
				<Row {...active_slice.props} />
				<Editor {...gallery_editor.props} />
			</div>
			<div class="flex flex-col space-y-3">
				<Match {...gallery_match.props} />

				<Filter {...gallery_filter.props} />
				<Sort {...gallery_sort.props} />
			</div>
		</div>
		<Gallery {...gallery.props} />
	</div>
</div>

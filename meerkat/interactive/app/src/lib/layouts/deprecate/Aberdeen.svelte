<script>
	import Plot from '$lib/component/plot/Plot.svelte';
	import Gallery from '$lib/component/gallery/Gallery.svelte';
	import Match from '$lib/component/match/Match.svelte';
	import Editor from '$lib/component/editor/Editor.svelte';
	import Choice from '$lib/component/choice/Choice.svelte';

	export let components;

	let gallery = components.gallery;
	let gallery_match = components.match;
	let editor = components.editor;
	let plot = components.plot;
	let x_choice = components.x_choice;
	let y_choice = components.y_choice;
</script>

<div class="grid grid-cols-[1fr_2fr] gap-x-6">
	<div class="grid grid-rows-[1fr_auto] gap-16 py-6">
		<Plot {...plot.props} />
        <div class="grid grid-cols-[auto_auto] gap-4 pl-20">
			<Choice {...x_choice.props} />
			<Choice {...y_choice.props} />
		</div>
	</div>
	<div class="h-screen grid grid-rows-[auto_1fr] gap-4 py-6 px-6">
		<div class="grid grid-cols-[auto_auto] gap-4">
			<Match {...gallery_match.props} />
			<Editor {...editor.props} />
		</div>
        <Gallery {...gallery.props} />

	</div>
</div>

<script lang="ts">
	export let component_id: string;
	export let name: string;
	export let path: string;
	export let props: any;

	import Button from '$lib/component/button/Button.svelte';
	import Choice from '$lib/component/choice/Choice.svelte';
	import Discover from '$lib/component/discover/Discover.svelte';
	import Document from '$lib/component/document/Document.svelte';
	import Editor from '$lib/component/editor/Editor.svelte';
	import Filter from '$lib/component/filter/Filter.svelte';
	import Gallery from '$lib/component/gallery/Gallery.svelte';
	import Markdown from '$lib/component/markdown/Markdown.svelte';
	import Plot from '$lib/component/plot/Plot.svelte';
	import Match from '$lib/component/match/Match.svelte';
	import Row from '$lib/component/row/Row.svelte';
	import SimpleSearchBar from '$lib/component/simplesearchbar/SimpleSearchBar.svelte';
	import SliceByCard from '$lib/component/slicebycards/SliceByCard.svelte';
	import SliceByCards from '$lib/component/slicebycards/SliceByCards.svelte';
	import Sort from '$lib/component/sort/Sort.svelte';
	import Stats from '$lib/component/stats/Stats.svelte';
	import StatsLabeler from '$lib/component/stats_labeler/StatsLabeler.svelte';
	import Table from '$lib/component/table/Table.svelte';
	import Tabs from '$lib/component/tabs/Tabs.svelte';
	import Textbox from '$lib/component/textbox/Textbox.svelte';
	
	let all_components = {
		'Button': Button,
		'Choice': Choice,
		'Discover': Discover,
		'Document': Document,
		'Editor': Editor,
		'Filter': Filter,
		'Gallery': Gallery,
		'Markdown': Markdown,
		'Match': Match,
		'Plot': Plot,
		'Row': Row,
		'SimpleSearchBar': SimpleSearchBar,
		'SliceByCard': SliceByCard,
		'SliceByCards': SliceByCards,
		'Sort': Sort,
		'Stats': Stats,
		'StatsLabeler': StatsLabeler,
		'Table': Table,
		'Tabs': Tabs,
		'Textbox': Textbox,
	};
</script>

<!-- {#await import(path)}
	Loading {name} component.
{:then Component}
	<svelte:component this={Component.default} {...props} />
{:catch error}
	{error}
{/await} -->

<svelte:component this={all_components[name]} {...props} />

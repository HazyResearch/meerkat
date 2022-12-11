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
	import MultiSelect from '$lib/component/multiselect/MultiSelect.svelte';
	import Plot from '$lib/component/plot/Plot.svelte';
	import Match from '$lib/component/match/Match.svelte';
	import Row from '$lib/component/row/Row.svelte';
	import SliceByCard from '$lib/component/slicebycards/SliceByCard.svelte';
	import SliceByCards from '$lib/component/slicebycards/SliceByCards.svelte';
	import Sort from '$lib/component/sort/Sort.svelte';
	import Stats from '$lib/component/stats/Stats.svelte';
	import StatsLabeler from '$lib/component/stats_labeler/StatsLabeler.svelte';
	import Table from '$lib/component/table/Table.svelte';
	import Tabs from '$lib/component/tabs/Tabs.svelte';
	import Text from '$lib/component/text/Text.svelte';

	import AutoLayout from '$lib/layouts/AutoLayout.svelte';
	import Div from '$lib/layouts/Div.svelte';
	import Flex from '$lib/layouts/Flex.svelte';
	import Grid from '$lib/layouts/Grid.svelte';

	import Image from './cell/image/Image.svelte';
	import Code from './cell/code/Code.svelte';
	import CodeDisplay from '$lib/component/codedisplay/CodeDisplay.svelte';

	let all_components = {
		Button: Button,
		Choice: Choice,
		Discover: Discover,
		Document: Document,
		Editor: Editor,
		Filter: Filter,
		Gallery: Gallery,
		Markdown: Markdown,
		Match: Match,
		MultiSelect: MultiSelect,
		Plot: Plot,
		Row: Row,
		SliceByCard: SliceByCard,
		SliceByCards: SliceByCards,
		Sort: Sort,
		Stats: Stats,
		StatsLabeler: StatsLabeler,
		Table: Table,
		Tabs: Tabs,
		Text: Text,

		// Layout components
		AutoLayout: AutoLayout,
		Div: Div,
		Flex: Flex,
		Grid: Grid,

		Image: Image,
		Code: Code,
		CodeDisplay: CodeDisplay
	};
</script>

{#if name in all_components}
	<svelte:component this={all_components[name]} {...props} />
{:else}
	{#await import(path)}
		Loading {name} component.
	{:then Component}
		<svelte:component this={Component.default} {...props} />
		<!-- {:catch error}
		{error} -->
	{/await}
{/if}

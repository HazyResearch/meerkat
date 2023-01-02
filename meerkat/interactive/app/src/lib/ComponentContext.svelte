<script lang="ts">
    import { setContext } from 'svelte/internal';
    
    import CarbonAccordion from '$lib/wrappers/__carbon/Accordion.svelte';
    import CarbonAccordionItem from '$lib/wrappers/__carbon/AccordionItem.svelte';
    import CarbonAccordionSkeleton from '$lib/wrappers/__carbon/AccordionSkeleton.svelte';
    import CarbonButton from '$lib/wrappers/__carbon/Button.svelte';
    import MeerkatAutoLayout from '$lib/wrappers/__meerkat/AutoLayout.svelte';
    import MeerkatButton from '$lib/wrappers/__meerkat/Button.svelte';
    import MeerkatChoice from '$lib/wrappers/__meerkat/Choice.svelte';
    import MeerkatCode from '$lib/wrappers/__meerkat/Code.svelte';
    import MeerkatCodeDisplay from '$lib/wrappers/__meerkat/CodeDisplay.svelte';
    import MeerkatFlex from '$lib/wrappers/__meerkat/Flex.svelte';
    import MeerkatDiscover from '$lib/wrappers/__meerkat/Discover.svelte';
    import MeerkatDiv from '$lib/wrappers/__meerkat/Div.svelte';
    import MeerkatDocument from '$lib/wrappers/__meerkat/Document.svelte';
    import MeerkatEditor from '$lib/wrappers/__meerkat/Editor.svelte';
    import MeerkatFilter from '$lib/wrappers/__meerkat/Filter.svelte';
    import MeerkatGallery from '$lib/wrappers/__meerkat/Gallery.svelte';
    import MeerkatGrid from '$lib/wrappers/__meerkat/Grid.svelte';
    import MeerkatImage from '$lib/wrappers/__meerkat/Image.svelte';
    import MeerkatMarkdown from '$lib/wrappers/__meerkat/Markdown.svelte';
    import MeerkatMatch from '$lib/wrappers/__meerkat/Match.svelte';
    import MeerkatMultiSelect from '$lib/wrappers/__meerkat/MultiSelect.svelte';
    import MeerkatPlot from '$lib/wrappers/__meerkat/Plot.svelte';
    import MeerkatRow from '$lib/wrappers/__meerkat/Row.svelte';
    import MeerkatSliceByCards from '$lib/wrappers/__meerkat/SliceByCards.svelte';
    import MeerkatSort from '$lib/wrappers/__meerkat/Sort.svelte';
    import MeerkatStats from '$lib/wrappers/__meerkat/Stats.svelte';
    import MeerkatStatsLabeler from '$lib/wrappers/__meerkat/StatsLabeler.svelte';
    import MeerkatTable from '$lib/wrappers/__meerkat/Table.svelte';
    import MeerkatTabs from '$lib/wrappers/__meerkat/Tabs.svelte';
    import MeerkatText from '$lib/wrappers/__meerkat/Text.svelte';
    import MeerkatTextbox from '$lib/wrappers/__meerkat/Textbox.svelte';
    import SvelteuiButton from '$lib/wrappers/__svelteui/Button.svelte';
    
    
    $: setContext("Components", {
        "CarbonAccordion": CarbonAccordion,
        "CarbonAccordionItem": CarbonAccordionItem,
        "CarbonAccordionSkeleton": CarbonAccordionSkeleton,
        "CarbonButton": CarbonButton,
        "MeerkatAutoLayout": MeerkatAutoLayout,
        "MeerkatButton": MeerkatButton,
        "MeerkatChoice": MeerkatChoice,
        "MeerkatCode": MeerkatCode,
        "MeerkatCodeDisplay": MeerkatCodeDisplay,
        "MeerkatColumnLayout": MeerkatFlex,
        "MeerkatDiscover": MeerkatDiscover,
        "MeerkatDiv": MeerkatDiv,
        "MeerkatDocument": MeerkatDocument,
        "MeerkatEditor": MeerkatEditor,
        "MeerkatFilter": MeerkatFilter,
        "MeerkatFlex": MeerkatFlex,
        "MeerkatGallery": MeerkatGallery,
        "MeerkatGrid": MeerkatGrid,
        "MeerkatImage": MeerkatImage,
        "MeerkatMarkdown": MeerkatMarkdown,
        "MeerkatMatch": MeerkatMatch,
        "MeerkatMultiSelect": MeerkatMultiSelect,
        "MeerkatPlot": MeerkatPlot,
        "MeerkatRow": MeerkatRow,
        "MeerkatRowLayout": MeerkatGrid,
        "MeerkatSliceByCards": MeerkatSliceByCards,
        "MeerkatSort": MeerkatSort,
        "MeerkatStats": MeerkatStats,
        "MeerkatStatsLabeler": MeerkatStatsLabeler,
        "MeerkatTable": MeerkatTable,
        "MeerkatTabs": MeerkatTabs,
        "MeerkatText": MeerkatText,
        "MeerkatTextbox": MeerkatTextbox,
        "SvelteuiButton": SvelteuiButton,
        
    })
</script>

<slot />
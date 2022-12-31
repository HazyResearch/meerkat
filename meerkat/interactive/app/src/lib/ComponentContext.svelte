<script lang="ts">
    import { setContext } from 'svelte/internal';
    
    import CarbonAccordion from 'src/lib/wrappers/__carbon/Accordion.svelte';
    import CarbonAccordionItem from 'src/lib/wrappers/__carbon/AccordionItem.svelte';
    import CarbonAccordionSkeleton from 'src/lib/wrappers/__carbon/AccordionSkeleton.svelte';
    import CarbonButton from 'src/lib/wrappers/__carbon/Button.svelte';
    import CustomExampleComponent from 'src/lib/wrappers/__custom/ExampleComponent.svelte';
    import MeerkatAutoLayout from 'src/lib/wrappers/__meerkat/AutoLayout.svelte';
    import MeerkatButton from 'src/lib/wrappers/__meerkat/Button.svelte';
    import MeerkatChoice from 'src/lib/wrappers/__meerkat/Choice.svelte';
    import MeerkatCode from 'src/lib/wrappers/__meerkat/Code.svelte';
    import MeerkatCodeDisplay from 'src/lib/wrappers/__meerkat/CodeDisplay.svelte';
    import MeerkatFlex from 'src/lib/wrappers/__meerkat/Flex.svelte';
    import MeerkatDiscover from 'src/lib/wrappers/__meerkat/Discover.svelte';
    import MeerkatDiv from 'src/lib/wrappers/__meerkat/Div.svelte';
    import MeerkatDocument from 'src/lib/wrappers/__meerkat/Document.svelte';
    import MeerkatEditor from 'src/lib/wrappers/__meerkat/Editor.svelte';
    import MeerkatFilter from 'src/lib/wrappers/__meerkat/Filter.svelte';
    import MeerkatGallery from 'src/lib/wrappers/__meerkat/Gallery.svelte';
    import MeerkatGrid from 'src/lib/wrappers/__meerkat/Grid.svelte';
    import MeerkatImage from 'src/lib/wrappers/__meerkat/Image.svelte';
    import MeerkatMarkdown from 'src/lib/wrappers/__meerkat/Markdown.svelte';
    import MeerkatMatch from 'src/lib/wrappers/__meerkat/Match.svelte';
    import MeerkatMultiSelect from 'src/lib/wrappers/__meerkat/MultiSelect.svelte';
    import MeerkatPlot from 'src/lib/wrappers/__meerkat/Plot.svelte';
    import MeerkatRow from 'src/lib/wrappers/__meerkat/Row.svelte';
    import MeerkatSliceByCards from 'src/lib/wrappers/__meerkat/SliceByCards.svelte';
    import MeerkatSlotGrid from 'src/lib/wrappers/__meerkat/SlotGrid.svelte';
    import MeerkatSort from 'src/lib/wrappers/__meerkat/Sort.svelte';
    import MeerkatStats from 'src/lib/wrappers/__meerkat/Stats.svelte';
    import MeerkatStatsLabeler from 'src/lib/wrappers/__meerkat/StatsLabeler.svelte';
    import MeerkatTable from 'src/lib/wrappers/__meerkat/Table.svelte';
    import MeerkatTabs from 'src/lib/wrappers/__meerkat/Tabs.svelte';
    import MeerkatText from 'src/lib/wrappers/__meerkat/Text.svelte';
    import MeerkatTextbox from 'src/lib/wrappers/__meerkat/Textbox.svelte';
    import SvelteuiButton from 'src/lib/wrappers/__svelteui/Button.svelte';
    
    
    $: setContext("Components", {
        "CarbonAccordion": CarbonAccordion,
        "CarbonAccordionItem": CarbonAccordionItem,
        "CarbonAccordionSkeleton": CarbonAccordionSkeleton,
        "CarbonButton": CarbonButton,
        "CustomExampleComponent": CustomExampleComponent,
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
        "MeerkatSlotGrid": MeerkatSlotGrid,
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
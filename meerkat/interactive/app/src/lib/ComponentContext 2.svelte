<script lang="ts">
    import { setContext } from 'svelte/internal';
    
    import FlowbiteA from '$lib/wrappers/__flowbite/A.svelte';
    import FlowbiteAccordion from '$lib/wrappers/__flowbite/Accordion.svelte';
    import FlowbiteAccordionItem from '$lib/wrappers/__flowbite/AccordionItem.svelte';
    import FlowbiteActivity from '$lib/wrappers/__flowbite/Activity.svelte';
    import FlowbiteActivityItem from '$lib/wrappers/__flowbite/ActivityItem.svelte';
    import FlowbiteAdvancedRating from '$lib/wrappers/__flowbite/AdvancedRating.svelte';
    import FlowbiteAlert from '$lib/wrappers/__flowbite/Alert.svelte';
    import FlowbiteArrowKeyDown from '$lib/wrappers/__flowbite/ArrowKeyDown.svelte';
    import FlowbiteArrowKeyLeft from '$lib/wrappers/__flowbite/ArrowKeyLeft.svelte';
    import FlowbiteArrowKeyRight from '$lib/wrappers/__flowbite/ArrowKeyRight.svelte';
    import FlowbiteArrowKeyUp from '$lib/wrappers/__flowbite/ArrowKeyUp.svelte';
    import FlowbiteAvatar from '$lib/wrappers/__flowbite/Avatar.svelte';
    import FlowbiteBadge from '$lib/wrappers/__flowbite/Badge.svelte';
    import FlowbiteBlockquote from '$lib/wrappers/__flowbite/Blockquote.svelte';
    import FlowbiteBreadcrumb from '$lib/wrappers/__flowbite/Breadcrumb.svelte';
    import FlowbiteBreadcrumbItem from '$lib/wrappers/__flowbite/BreadcrumbItem.svelte';
    import FlowbiteButton from '$lib/wrappers/__flowbite/Button.svelte';
    import FlowbiteButtonGroup from '$lib/wrappers/__flowbite/ButtonGroup.svelte';
    import FlowbiteCard from '$lib/wrappers/__flowbite/Card.svelte';
    import FlowbiteCarousel from '$lib/wrappers/__flowbite/Carousel.svelte';
    import FlowbiteCarouselTransition from '$lib/wrappers/__flowbite/CarouselTransition.svelte';
    import FlowbiteCheckbox from '$lib/wrappers/__flowbite/Checkbox.svelte';
    import FlowbiteDarkMode from '$lib/wrappers/__flowbite/DarkMode.svelte';
    import FlowbiteDescriptionList from '$lib/wrappers/__flowbite/DescriptionList.svelte';
    import FlowbiteDrawer from '$lib/wrappers/__flowbite/Drawer.svelte';
    import FlowbiteDropdown from '$lib/wrappers/__flowbite/Dropdown.svelte';
    import FlowbiteDropdownDivider from '$lib/wrappers/__flowbite/DropdownDivider.svelte';
    import FlowbiteDropdownHeader from '$lib/wrappers/__flowbite/DropdownHeader.svelte';
    import FlowbiteDropdownItem from '$lib/wrappers/__flowbite/DropdownItem.svelte';
    import FlowbiteFloatingLabelInput from '$lib/wrappers/__flowbite/FloatingLabelInput.svelte';
    import FlowbiteFooter from '$lib/wrappers/__flowbite/Footer.svelte';
    import FlowbiteFooterBrand from '$lib/wrappers/__flowbite/FooterBrand.svelte';
    import FlowbiteFooterCopyright from '$lib/wrappers/__flowbite/FooterCopyright.svelte';
    import FlowbiteFooterIcon from '$lib/wrappers/__flowbite/FooterIcon.svelte';
    import FlowbiteFooterLink from '$lib/wrappers/__flowbite/FooterLink.svelte';
    import FlowbiteFooterLinkGroup from '$lib/wrappers/__flowbite/FooterLinkGroup.svelte';
    import FlowbiteGroup from '$lib/wrappers/__flowbite/Group.svelte';
    import FlowbiteGroupItem from '$lib/wrappers/__flowbite/GroupItem.svelte';
    import FlowbiteHeading from '$lib/wrappers/__flowbite/Heading.svelte';
    import FlowbiteHr from '$lib/wrappers/__flowbite/Hr.svelte';
    import FlowbiteIndicator from '$lib/wrappers/__flowbite/Indicator.svelte';
    import FlowbiteKbd from '$lib/wrappers/__flowbite/Kbd.svelte';
    import FlowbiteLayout from '$lib/wrappers/__flowbite/Layout.svelte';
    import FlowbiteLi from '$lib/wrappers/__flowbite/Li.svelte';
    import FlowbiteList from '$lib/wrappers/__flowbite/List.svelte';
    import FlowbiteMark from '$lib/wrappers/__flowbite/Mark.svelte';
    import FlowbiteMegaMenu from '$lib/wrappers/__flowbite/MegaMenu.svelte';
    import FlowbiteModal from '$lib/wrappers/__flowbite/Modal.svelte';
    import FlowbiteNavBrand from '$lib/wrappers/__flowbite/NavBrand.svelte';
    import FlowbiteNavLi from '$lib/wrappers/__flowbite/NavLi.svelte';
    import FlowbiteNavUl from '$lib/wrappers/__flowbite/NavUl.svelte';
    import FlowbiteNavbar from '$lib/wrappers/__flowbite/Navbar.svelte';
    import FlowbiteP from '$lib/wrappers/__flowbite/P.svelte';
    import FlowbitePagination from '$lib/wrappers/__flowbite/Pagination.svelte';
    import FlowbitePaginationItem from '$lib/wrappers/__flowbite/PaginationItem.svelte';
    import FlowbitePopover from '$lib/wrappers/__flowbite/Popover.svelte';
    import FlowbiteProgressbar from '$lib/wrappers/__flowbite/Progressbar.svelte';
    import FlowbiteRadio from '$lib/wrappers/__flowbite/Radio.svelte';
    import FlowbiteRange from '$lib/wrappers/__flowbite/Range.svelte';
    import FlowbiteRating from '$lib/wrappers/__flowbite/Rating.svelte';
    import FlowbiteRatingComment from '$lib/wrappers/__flowbite/RatingComment.svelte';
    import FlowbiteReview from '$lib/wrappers/__flowbite/Review.svelte';
    import FlowbiteScoreRating from '$lib/wrappers/__flowbite/ScoreRating.svelte';
    import FlowbiteSearch from '$lib/wrappers/__flowbite/Search.svelte';
    import FlowbiteSecondary from '$lib/wrappers/__flowbite/Secondary.svelte';
    import FlowbiteSelect from '$lib/wrappers/__flowbite/Select.svelte';
    import FlowbiteSidebar from '$lib/wrappers/__flowbite/Sidebar.svelte';
    import FlowbiteSidebarBrand from '$lib/wrappers/__flowbite/SidebarBrand.svelte';
    import FlowbiteSidebarCta from '$lib/wrappers/__flowbite/SidebarCta.svelte';
    import FlowbiteSidebarDropdownItem from '$lib/wrappers/__flowbite/SidebarDropdownItem.svelte';
    import FlowbiteSidebarDropdownWrapper from '$lib/wrappers/__flowbite/SidebarDropdownWrapper.svelte';
    import FlowbiteSidebarGroup from '$lib/wrappers/__flowbite/SidebarGroup.svelte';
    import FlowbiteSidebarItem from '$lib/wrappers/__flowbite/SidebarItem.svelte';
    import FlowbiteSidebarWrapper from '$lib/wrappers/__flowbite/SidebarWrapper.svelte';
    import FlowbiteSpan from '$lib/wrappers/__flowbite/Span.svelte';
    import FlowbiteSpeedDial from '$lib/wrappers/__flowbite/SpeedDial.svelte';
    import FlowbiteSpeedDialButton from '$lib/wrappers/__flowbite/SpeedDialButton.svelte';
    import FlowbiteSpinner from '$lib/wrappers/__flowbite/Spinner.svelte';
    import FlowbiteTabItem from '$lib/wrappers/__flowbite/TabItem.svelte';
    import FlowbiteTable from '$lib/wrappers/__flowbite/Table.svelte';
    import FlowbiteTableBodyCell from '$lib/wrappers/__flowbite/TableBodyCell.svelte';
    import FlowbiteTableBodyRow from '$lib/wrappers/__flowbite/TableBodyRow.svelte';
    import FlowbiteTableHead from '$lib/wrappers/__flowbite/TableHead.svelte';
    import FlowbiteTableSearch from '$lib/wrappers/__flowbite/TableSearch.svelte';
    import FlowbiteTabs from '$lib/wrappers/__flowbite/Tabs.svelte';
    import FlowbiteTextarea from '$lib/wrappers/__flowbite/Textarea.svelte';
    import FlowbiteTimeline from '$lib/wrappers/__flowbite/Timeline.svelte';
    import FlowbiteTimelineHorizontal from '$lib/wrappers/__flowbite/TimelineHorizontal.svelte';
    import FlowbiteTimelineItem from '$lib/wrappers/__flowbite/TimelineItem.svelte';
    import FlowbiteTimelineItemHorizontal from '$lib/wrappers/__flowbite/TimelineItemHorizontal.svelte';
    import FlowbiteToast from '$lib/wrappers/__flowbite/Toast.svelte';
    import FlowbiteToggle from '$lib/wrappers/__flowbite/Toggle.svelte';
    import FlowbiteTooltip from '$lib/wrappers/__flowbite/Tooltip.svelte';
    import Htmla from '$lib/wrappers/__html/a.svelte';
    import Htmlbutton from '$lib/wrappers/__html/button.svelte';
    import Htmldiv from '$lib/wrappers/__html/div.svelte';
    import Htmlform from '$lib/wrappers/__html/form.svelte';
    import Htmlh1 from '$lib/wrappers/__html/h1.svelte';
    import Htmlh2 from '$lib/wrappers/__html/h2.svelte';
    import Htmlh3 from '$lib/wrappers/__html/h3.svelte';
    import Htmlh4 from '$lib/wrappers/__html/h4.svelte';
    import Htmlh5 from '$lib/wrappers/__html/h5.svelte';
    import Htmlh6 from '$lib/wrappers/__html/h6.svelte';
    import Htmlli from '$lib/wrappers/__html/li.svelte';
    import Htmlol from '$lib/wrappers/__html/ol.svelte';
    import Htmlp from '$lib/wrappers/__html/p.svelte';
    import Htmlpath from '$lib/wrappers/__html/path.svelte';
    import Htmlselect from '$lib/wrappers/__html/select.svelte';
    import Htmlspan from '$lib/wrappers/__html/span.svelte';
    import Htmlsvg from '$lib/wrappers/__html/svg.svelte';
    import Htmltable from '$lib/wrappers/__html/table.svelte';
    import Htmltbody from '$lib/wrappers/__html/tbody.svelte';
    import Htmltd from '$lib/wrappers/__html/td.svelte';
    import Htmltextarea from '$lib/wrappers/__html/textarea.svelte';
    import Htmlth from '$lib/wrappers/__html/th.svelte';
    import Htmlthead from '$lib/wrappers/__html/thead.svelte';
    import Htmltr from '$lib/wrappers/__html/tr.svelte';
    import Htmlul from '$lib/wrappers/__html/ul.svelte';
    import MeerkatButton from '$lib/wrappers/__meerkat/Button.svelte';
    import MeerkatMarkdown from '$lib/wrappers/__meerkat/Markdown.svelte';
    import MeerkatChangeList from '$lib/wrappers/__meerkat/ChangeList.svelte';
    import MeerkatChat from '$lib/wrappers/__meerkat/Chat.svelte';
    import MeerkatCheckbox from '$lib/wrappers/__meerkat/Checkbox.svelte';
    import MeerkatCode from '$lib/wrappers/__meerkat/Code.svelte';
    import MeerkatCodeCell from '$lib/wrappers/__meerkat/CodeCell.svelte';
    import MeerkatDiscover from '$lib/wrappers/__meerkat/Discover.svelte';
    import MeerkatDocument from '$lib/wrappers/__meerkat/Document.svelte';
    import MeerkatFMFilter from '$lib/wrappers/__meerkat/FMFilter.svelte';
    import MeerkatFileUpload from '$lib/wrappers/__meerkat/FileUpload.svelte';
    import MeerkatFilter from '$lib/wrappers/__meerkat/Filter.svelte';
    import MeerkatGallery from '$lib/wrappers/__meerkat/Gallery.svelte';
    import MeerkatGlobalStats from '$lib/wrappers/__meerkat/GlobalStats.svelte';
    import MeerkatImage from '$lib/wrappers/__meerkat/Image.svelte';
    import MeerkatJson from '$lib/wrappers/__meerkat/Json.svelte';
    import MeerkatMatch from '$lib/wrappers/__meerkat/Match.svelte';
    import MeerkatMultiSelect from '$lib/wrappers/__meerkat/MultiSelect.svelte';
    import MeerkatPlot from '$lib/wrappers/__meerkat/Plot.svelte';
    import MeerkatProgress from '$lib/wrappers/__meerkat/Progress.svelte';
    import MeerkatPut from '$lib/wrappers/__meerkat/Put.svelte';
    import MeerkatRadio from '$lib/wrappers/__meerkat/Radio.svelte';
    import MeerkatRadioGroup from '$lib/wrappers/__meerkat/RadioGroup.svelte';
    import MeerkatRawHTML from '$lib/wrappers/__meerkat/RawHTML.svelte';
    import MeerkatRow from '$lib/wrappers/__meerkat/Row.svelte';
    import MeerkatScalar from '$lib/wrappers/__meerkat/Scalar.svelte';
    import MeerkatSelect from '$lib/wrappers/__meerkat/Select.svelte';
    import MeerkatSliceByCards from '$lib/wrappers/__meerkat/SliceByCards.svelte';
    import MeerkatSlider from '$lib/wrappers/__meerkat/Slider.svelte';
    import MeerkatSort from '$lib/wrappers/__meerkat/Sort.svelte';
    import MeerkatStats from '$lib/wrappers/__meerkat/Stats.svelte';
    import MeerkatTable from '$lib/wrappers/__meerkat/Table.svelte';
    import MeerkatTabs from '$lib/wrappers/__meerkat/Tabs.svelte';
    import MeerkatText from '$lib/wrappers/__meerkat/Text.svelte';
    import MeerkatTextbox from '$lib/wrappers/__meerkat/Textbox.svelte';
    import MeerkatToggle from '$lib/wrappers/__meerkat/Toggle.svelte';
    import MeerkatVega from '$lib/wrappers/__meerkat/Vega.svelte';
    import MeerkatWebsite from '$lib/wrappers/__meerkat/Website.svelte';
    import PlotlyBarPlot from '$lib/wrappers/__plotly/BarPlot.svelte';
    import PlotlyHistogram from '$lib/wrappers/__plotly/Histogram.svelte';
    import PlotlyPlot from '$lib/wrappers/__plotly/Plot.svelte';
    
    
    $: setContext("Components", {
        "FlowbiteA": FlowbiteA,
        "FlowbiteAccordion": FlowbiteAccordion,
        "FlowbiteAccordionItem": FlowbiteAccordionItem,
        "FlowbiteActivity": FlowbiteActivity,
        "FlowbiteActivityItem": FlowbiteActivityItem,
        "FlowbiteAdvancedRating": FlowbiteAdvancedRating,
        "FlowbiteAlert": FlowbiteAlert,
        "FlowbiteArrowKeyDown": FlowbiteArrowKeyDown,
        "FlowbiteArrowKeyLeft": FlowbiteArrowKeyLeft,
        "FlowbiteArrowKeyRight": FlowbiteArrowKeyRight,
        "FlowbiteArrowKeyUp": FlowbiteArrowKeyUp,
        "FlowbiteAvatar": FlowbiteAvatar,
        "FlowbiteBadge": FlowbiteBadge,
        "FlowbiteBlockquote": FlowbiteBlockquote,
        "FlowbiteBreadcrumb": FlowbiteBreadcrumb,
        "FlowbiteBreadcrumbItem": FlowbiteBreadcrumbItem,
        "FlowbiteButton": FlowbiteButton,
        "FlowbiteButtonGroup": FlowbiteButtonGroup,
        "FlowbiteCard": FlowbiteCard,
        "FlowbiteCarousel": FlowbiteCarousel,
        "FlowbiteCarouselTransition": FlowbiteCarouselTransition,
        "FlowbiteCheckbox": FlowbiteCheckbox,
        "FlowbiteDarkMode": FlowbiteDarkMode,
        "FlowbiteDescriptionList": FlowbiteDescriptionList,
        "FlowbiteDrawer": FlowbiteDrawer,
        "FlowbiteDropdown": FlowbiteDropdown,
        "FlowbiteDropdownDivider": FlowbiteDropdownDivider,
        "FlowbiteDropdownHeader": FlowbiteDropdownHeader,
        "FlowbiteDropdownItem": FlowbiteDropdownItem,
        "FlowbiteFloatingLabelInput": FlowbiteFloatingLabelInput,
        "FlowbiteFooter": FlowbiteFooter,
        "FlowbiteFooterBrand": FlowbiteFooterBrand,
        "FlowbiteFooterCopyright": FlowbiteFooterCopyright,
        "FlowbiteFooterIcon": FlowbiteFooterIcon,
        "FlowbiteFooterLink": FlowbiteFooterLink,
        "FlowbiteFooterLinkGroup": FlowbiteFooterLinkGroup,
        "FlowbiteGroup": FlowbiteGroup,
        "FlowbiteGroupItem": FlowbiteGroupItem,
        "FlowbiteHeading": FlowbiteHeading,
        "FlowbiteHr": FlowbiteHr,
        "FlowbiteIndicator": FlowbiteIndicator,
        "FlowbiteKbd": FlowbiteKbd,
        "FlowbiteLayout": FlowbiteLayout,
        "FlowbiteLi": FlowbiteLi,
        "FlowbiteList": FlowbiteList,
        "FlowbiteMark": FlowbiteMark,
        "FlowbiteMegaMenu": FlowbiteMegaMenu,
        "FlowbiteModal": FlowbiteModal,
        "FlowbiteNavBrand": FlowbiteNavBrand,
        "FlowbiteNavLi": FlowbiteNavLi,
        "FlowbiteNavUl": FlowbiteNavUl,
        "FlowbiteNavbar": FlowbiteNavbar,
        "FlowbiteP": FlowbiteP,
        "FlowbitePagination": FlowbitePagination,
        "FlowbitePaginationItem": FlowbitePaginationItem,
        "FlowbitePopover": FlowbitePopover,
        "FlowbiteProgressbar": FlowbiteProgressbar,
        "FlowbiteRadio": FlowbiteRadio,
        "FlowbiteRange": FlowbiteRange,
        "FlowbiteRating": FlowbiteRating,
        "FlowbiteRatingComment": FlowbiteRatingComment,
        "FlowbiteReview": FlowbiteReview,
        "FlowbiteScoreRating": FlowbiteScoreRating,
        "FlowbiteSearch": FlowbiteSearch,
        "FlowbiteSecondary": FlowbiteSecondary,
        "FlowbiteSelect": FlowbiteSelect,
        "FlowbiteSidebar": FlowbiteSidebar,
        "FlowbiteSidebarBrand": FlowbiteSidebarBrand,
        "FlowbiteSidebarCta": FlowbiteSidebarCta,
        "FlowbiteSidebarDropdownItem": FlowbiteSidebarDropdownItem,
        "FlowbiteSidebarDropdownWrapper": FlowbiteSidebarDropdownWrapper,
        "FlowbiteSidebarGroup": FlowbiteSidebarGroup,
        "FlowbiteSidebarItem": FlowbiteSidebarItem,
        "FlowbiteSidebarWrapper": FlowbiteSidebarWrapper,
        "FlowbiteSpan": FlowbiteSpan,
        "FlowbiteSpeedDial": FlowbiteSpeedDial,
        "FlowbiteSpeedDialButton": FlowbiteSpeedDialButton,
        "FlowbiteSpinner": FlowbiteSpinner,
        "FlowbiteTabItem": FlowbiteTabItem,
        "FlowbiteTable": FlowbiteTable,
        "FlowbiteTableBodyCell": FlowbiteTableBodyCell,
        "FlowbiteTableBodyRow": FlowbiteTableBodyRow,
        "FlowbiteTableHead": FlowbiteTableHead,
        "FlowbiteTableSearch": FlowbiteTableSearch,
        "FlowbiteTabs": FlowbiteTabs,
        "FlowbiteTextarea": FlowbiteTextarea,
        "FlowbiteTimeline": FlowbiteTimeline,
        "FlowbiteTimelineHorizontal": FlowbiteTimelineHorizontal,
        "FlowbiteTimelineItem": FlowbiteTimelineItem,
        "FlowbiteTimelineItemHorizontal": FlowbiteTimelineItemHorizontal,
        "FlowbiteToast": FlowbiteToast,
        "FlowbiteToggle": FlowbiteToggle,
        "FlowbiteTooltip": FlowbiteTooltip,
        "Htmla": Htmla,
        "Htmlbutton": Htmlbutton,
        "Htmldiv": Htmldiv,
        "Htmlflex": Htmldiv,
        "Htmlflexcol": Htmldiv,
        "Htmlform": Htmlform,
        "Htmlgrid": Htmldiv,
        "Htmlgridcols2": Htmldiv,
        "Htmlgridcols3": Htmldiv,
        "Htmlgridcols4": Htmldiv,
        "Htmlh1": Htmlh1,
        "Htmlh2": Htmlh2,
        "Htmlh3": Htmlh3,
        "Htmlh4": Htmlh4,
        "Htmlh5": Htmlh5,
        "Htmlh6": Htmlh6,
        "Htmlli": Htmlli,
        "Htmlol": Htmlol,
        "Htmlp": Htmlp,
        "Htmlpath": Htmlpath,
        "Htmlselect": Htmlselect,
        "Htmlspan": Htmlspan,
        "Htmlsvg": Htmlsvg,
        "Htmltable": Htmltable,
        "Htmltbody": Htmltbody,
        "Htmltd": Htmltd,
        "Htmltextarea": Htmltextarea,
        "Htmlth": Htmlth,
        "Htmlthead": Htmlthead,
        "Htmltr": Htmltr,
        "Htmlul": Htmlul,
        "MeerkatButton": MeerkatButton,
        "MeerkatCaption": MeerkatMarkdown,
        "MeerkatChangeList": MeerkatChangeList,
        "MeerkatChat": MeerkatChat,
        "MeerkatCheckbox": MeerkatCheckbox,
        "MeerkatCode": MeerkatCode,
        "MeerkatCodeCell": MeerkatCodeCell,
        "MeerkatDiscover": MeerkatDiscover,
        "MeerkatDocument": MeerkatDocument,
        "MeerkatFMFilter": MeerkatFMFilter,
        "MeerkatFileUpload": MeerkatFileUpload,
        "MeerkatFilter": MeerkatFilter,
        "MeerkatFilterCodeCell": MeerkatCodeCell,
        "MeerkatGallery": MeerkatGallery,
        "MeerkatGlobalStats": MeerkatGlobalStats,
        "MeerkatHeader": MeerkatMarkdown,
        "MeerkatImage": MeerkatImage,
        "MeerkatJson": MeerkatJson,
        "MeerkatMarkdown": MeerkatMarkdown,
        "MeerkatMatch": MeerkatMatch,
        "MeerkatMultiSelect": MeerkatMultiSelect,
        "MeerkatPlot": MeerkatPlot,
        "MeerkatProgress": MeerkatProgress,
        "MeerkatPut": MeerkatPut,
        "MeerkatRadio": MeerkatRadio,
        "MeerkatRadioGroup": MeerkatRadioGroup,
        "MeerkatRawHTML": MeerkatRawHTML,
        "MeerkatRow": MeerkatRow,
        "MeerkatScalar": MeerkatScalar,
        "MeerkatSelect": MeerkatSelect,
        "MeerkatSliceByCards": MeerkatSliceByCards,
        "MeerkatSlider": MeerkatSlider,
        "MeerkatSort": MeerkatSort,
        "MeerkatStats": MeerkatStats,
        "MeerkatSubheader": MeerkatMarkdown,
        "MeerkatTable": MeerkatTable,
        "MeerkatTabs": MeerkatTabs,
        "MeerkatText": MeerkatText,
        "MeerkatTextbox": MeerkatTextbox,
        "MeerkatTitle": MeerkatMarkdown,
        "MeerkatToggle": MeerkatToggle,
        "MeerkatVega": MeerkatVega,
        "MeerkatWebsite": MeerkatWebsite,
        "PlotlyBarPlot": PlotlyBarPlot,
        "PlotlyHistogram": PlotlyHistogram,
        "PlotlyPlot": PlotlyPlot,
        
    })
</script>

<slot />
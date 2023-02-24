<script lang="ts">
	import Footer from '$lib/Footer.svelte';
	import Nav from '$lib/Nav.svelte';

	import {
		Button,
		Chat,
		Checkbox,
		Code,
		BarPlot,
		Filter,
		Stats,
		Vega,
		RawHTML,
		RadioGroup,
		Text,
		Textbox,
		Slider,
		Radio,
		Gallery,
		Tabs,
		Image,
		Json,
		FileUpload,
		Select,
		Table,
		Toggle,
		Markdown,
		Document,
		Sort,
		Match,
		Plot
	} from '@meerkat-ml/meerkat';
	import {
		Label,
		Sidebar,
		SidebarBrand,
		SidebarDropdownItem,
		SidebarDropdownWrapper,
		SidebarGroup,
		SidebarItem,
		SidebarWrapper
	} from 'flowbite-svelte';
	import { setContext } from 'svelte';

	setContext('Components', {
		Text: Text
	});

	const messages = {
		columnInfos: [
			{
				name: 'id',
				type: 'number',
				cellComponent: 'Cell',
				cellProps: {},
				cellDataProp: 'data'
			},
			{
				name: 'message',
				type: 'string',
				cellComponent: 'Cell',
				cellProps: {},
				cellDataProp: 'data'
			},
			{
				name: 'name',
				type: 'string',
				cellComponent: 'Cell',
				cellProps: {},
				cellDataProp: 'data'
			},
			{
				name: 'time',
				type: 'string',
				cellComponent: 'Cell',
				cellProps: {},
				cellDataProp: 'data'
			},
			{
				name: 'sender',
				type: 'string',
				cellComponent: 'Cell',
				cellProps: {},
				cellDataProp: 'data'
			}
		],
		posidxs: [0, 1, 2, 3],
		rows: [
			[0, 'hello', 'chatbot', '2021-01-01 00:00:00'],
			[1, 'hi', 'user', '2021-01-01 00:00:00'],
			[2, 'how can i help', 'chatbot', '2021-01-01 00:00:00'],
			[3, 'im good thanks', 'user', '2021-01-01 00:00:00']
		],
		fullLength: 4,
		primaryKey: 'id'
	};
</script>

<div class="flex flex-col h-screen">
	<header class="fixed w-full top-0 z-50">
		<Nav page="components" />
	</header>
	<main class="mt-20 relative grow">
		<!-- Make a sidebar -->
		<div class="flex flex-row">
			<Sidebar>
				<SidebarWrapper>
                    <!-- <SidebarBrand label="Meerkat" /> -->
					<SidebarGroup>
						<SidebarDropdownWrapper label="The Basics">
							<SidebarDropdownItem label="Button" />
							<SidebarDropdownItem label="Checkbox" />
							<SidebarDropdownItem label="File Upload" />
							<SidebarDropdownItem label="Image" />
							<SidebarDropdownItem label="Json" />
							<SidebarDropdownItem label="Markdown" />
							<SidebarDropdownItem label="Slider" />
							<SidebarDropdownItem label="Toggle" />
							<SidebarDropdownItem label="Text" />
							<SidebarDropdownItem label="Textbox" />
							<SidebarDropdownItem label="Radio" />
							<SidebarDropdownItem label="Radio Group" />
							<SidebarDropdownItem label="Select" />
							<SidebarDropdownItem label="Tabs" />
							<SidebarDropdownItem label="Stats" />
						</SidebarDropdownWrapper>
						<SidebarDropdownWrapper label="Dataframe Components">
							<SidebarDropdownItem label="Bar Plot" />
							<SidebarDropdownItem label="Document" />
							<SidebarDropdownItem label="Filter" />
							<SidebarDropdownItem label="Gallery" />
							<SidebarDropdownItem label="Match" />
							<SidebarDropdownItem label="Plot" />
							<SidebarDropdownItem label="Sort" />
							<SidebarDropdownItem label="Table" />
						</SidebarDropdownWrapper>
						<SidebarItem label="Flowbite Components" />
						<SidebarItem label="HTML Tags" />
					</SidebarGroup>
				</SidebarWrapper>
			</Sidebar>

			<div>
				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Button</Label></div>
					<div class="w-20"><Button title="Hello" /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Checkbox</Label></div>
					<div class="w-fit"><Checkbox checked={true} /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Text</Label></div>
					<div class="w-fit"><Text data={'A quick brown fox jumped over a lazy dog.'} /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Textbox</Label></div>
					<div class="w-fit"><Textbox text="Write." /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Slider</Label></div>
					<div class="w-fit"><Slider min={0} max={10} /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Radio</Label></div>
					<div class="w-fit"><Radio /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>RawHTML</Label></div>
					<div class="w-fit">
						<RawHTML
							html={`
<div>
    <h1>Hello</h1>
    <p>World</p>
</div>
`}
						/>
					</div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>RadioGroup</Label></div>
					<div class="w-fit"><RadioGroup values={[1, 2, 3]} /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>FileUpload</Label></div>
					<div class="w-fit"><FileUpload files={[]} filenames={[]} contents={[]} /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Stats</Label></div>
					<div class="w-fit">
						<Stats data={{ 'Temperature (°C)': 13, 'Altitude (ft)': 12345 }} />
					</div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Toggle</Label></div>
					<div class="w-fit"><Toggle value={false} /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Code</Label></div>
					<Code body="import meerkat as mk" />
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Image</Label></div>
					<div class="w-fit"><Image data="http://placekitten.com/200/200" /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Markdown</Label></div>
					<div class="w-fit">
						<Markdown
							body={`
# Github Flavored Markdown
## Hello from Meerkat
### Table
| A | B |
|---|---|
| 1 | 2 |
`}
						/>
					</div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Json</Label></div>
					<div class="w-fit"><Json body={messages} /></div>
				</div>

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Tabs</Label></div>
					<div class="w-fit">
						<Tabs
							tabs={[
								{
									label: 'Tab 1',
									id: 'tab1',
									component: { name: 'Text', props: { data: 'This is a component in Tab 1' } }
								},
								{
									label: 'Tab 2',
									id: 'tab2',
									component: { name: 'Text', props: { data: 'This is a component in Tab 2' } }
								},
								{
									label: 'Tab 3',
									id: 'tab3',
									component: { name: 'Text', props: { data: 'This is a component in Tab 3' } }
								}
							]}
						/>
					</div>
				</div>

				<!-- <div class="flex flex-col items-center mb-4">
			<div class="flex flex-col items-center mt-4 mb-4"><Label>Gallery</Label></div>
			<Gallery />
		</div> -->

				<!-- <div class="flex flex-col items-center mb-4">
            <div class="flex flex-col items-center mt-4 mb-4"><Label>Table</Label></div>
            <Table df={{refId: "mock"}} />
        </div> -->

				<div class="flex flex-col items-center mb-4">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Select</Label></div>
					<Select
						values={[1, 2, 3, 4]}
						labels={['Option 1', 'Option 2', 'Option 3', 'Option 4']}
						value="1"
					/>
				</div>

				<div class="flex flex-col items-center mb-4 h-64">
					<div class="flex flex-col items-center mt-4 mb-4"><Label>Chat</Label></div>
					<Chat df={{ refId: 'mock' }} />
				</div>
			</div>
		</div>
	</main>
	<footer class="sticky bottom-0 z-50">
		<Footer />
	</footer>
</div>

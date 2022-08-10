<script lang="ts">
	import { create_column, type DataPanelSchema } from '$lib/api/datapanel';
	import { api_url } from '$lib/../routes/network/stores';
	import Select from 'svelte-select';

	export let datapanel_id: string;
	export let schema_promise: Promise<DataPanelSchema>;
	
	let search_box_text: string = '';
	let create_column_promise: Promise<string>;

	let on_search = async () => {
		create_column_promise = create_column($api_url, '0', search_box_text);
	};
	const onKeyPress = (e) => {
		if (e.charCode === 13) on_search();
	};

	const empty_items: Array<any> = [];
	let items_promise = schema_promise.then(
		(schema) => {
			return schema.columns.map((column) => {
				return {
					value: column.name,
					label: column.name,
				};
			});
		}
	);

	let favouriteFood = undefined;

	function handleSelect(event) {
		favouriteFood = event.detail;
	}

	function handleClear() {
		favouriteFood = undefined;
	}
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md">
	<div class="form-control w-full pl-10">
		<div class="input-group w-full flex items-center">
			<input
				type="text"
				bind:value={search_box_text}
				placeholder="Write some text to be matched..."
				class="input input-bordered w-3/5 h-10 px-3 rounded-md shadow-md"
				on:keypress={onKeyPress}
			/>
			<div class="text-slate-400 px-2"> against </div>

			<div class="themed">
				{#await items_promise}
					<Select 
						id="column" 
						placeholder="...a column."
						isWaiting={true}
						showIndicator={true}
					/>
				{:then items}
					<Select 
						id="column" 
						placeholder="...a column."
						{items} 
						showIndicator={true},
						listPlacement="auto",
						on:select={handleSelect} 
						on:clear={handleClear} />
				{/await}
			</div>
		</div>
	</div>
	{#await create_column_promise}
	<div class="h-full">Loading data...</div>
	{:then response_text}
		<div class="h-full">{response_text}</div>
	{/await}
</div>

<style>	
	/* 	
			CSS variables can be used to control theming.
			https://github.com/rob-balfre/svelte-select/blob/master/docs/theming_variables.md
	*/
	
	.themed {
		--itemPadding: 0.1rem;
		--itemColor: "#7c3aed"; 
		@apply rounded-md w-40 border-0;
		@apply z-[1000000];
	}
	.list-container {
		z-index: 2000;
		position: relative;
	}
</style>

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

<div class="bg-slate-100 py-10 my-10 rounded-lg">
	<div class="form-control w-full pl-10">
		<div class="input-group w-full">
			<input
				type="text"
				bind:value={search_box_text}
				placeholder="Search…"
				class="input input-bordered w-3/4 h-10 px-3"
				on:keypress={onKeyPress}
			/>
			<button class="btn btn-square" on:click={on_search}>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-6 w-6"
					fill="none"
					viewBox="0 0 24 24"
					stroke="currentColor"
					><path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
					/></svg
				>
			</button>
			<div class="flex">
				Searching against column
				<div class="w-20">
					{#await items_promise}
						<Select id="food" {empty_items} on:select={handleSelect} on:clear={handleClear} />
					{:then items}
						<Select id="food" {items} on:select={handleSelect} on:clear={handleClear} />
					{/await}
				</div>
			</div>

			{#await create_column_promise}
				<div class="h-full">Loading data...</div>
			{:then response_text}
				<div class="h-full">{response_text}</div>
			{/await}
		</div>
	</div>
</div>

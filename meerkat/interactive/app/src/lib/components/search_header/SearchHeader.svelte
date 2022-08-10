<script lang="ts">
	import { match, sort, type DataPanelSchema } from '$lib/api/datapanel';
	import { api_url } from '$lib/../routes/network/stores';
	import type { RefreshCallback } from '$lib/api/callbacks';
	import Select from 'svelte-select';

	export let base_datapanel_id: string;
	export let schema_promise: Promise<DataPanelSchema>;
	export let refresh_callback: RefreshCallback;
	
	let search_box_text: string = '';
	let search_promise: Promise<DataPanelSchema>;

	let on_search = async () => {
		console.log("on search: asdfasdfsafasf")
		let match_output: DataPanelSchema = await match($api_url, base_datapanel_id, search_box_text, "img");
		console.log("on search: Match complete")
		search_promise = sort($api_url, base_datapanel_id, match_output.columns[0].name)
		console.log("on search: Sort complete")
		search_promise.then(
			(schema: DataPanelSchema) => {
				refresh_callback(schema.id)
			}
		)
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

			{#await search_promise}
				<div class="h-full">Matching...</div>
			{:then items}
				<div class="h-full">Finished...</div>
			{/await}
		</div>
	</div>
</div>

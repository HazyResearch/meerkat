<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		let api_server_url = import.meta.env['VITE_API_URL'];
		const id = url.searchParams.get('id');
		const response = await fetch(`${api_server_url}/interface/${id}/config`);

		return {
			status: response.status,
			props: {
				config: response.ok && (await response.json())
			}
		};
	}
</script>

<script lang="ts">
	import { setContext } from 'svelte';
	import { writable, type Writable } from 'svelte/store';

	import type { SliceKey } from '$lib/api/sliceby';
	import StoreComponent from '$lib/component/StoreComponent.svelte';
	import { global_stores, meerkat_writable } from '$lib/shared/blanks/stores';
	import { get_request, modify, post } from '$lib/utils/requests';
	import { nestedMap } from '$lib/utils/tools';
	import type { EditTarget, Interface } from '$lib/utils/types';
	import { onMount } from 'svelte';

	let api_url = writable(import.meta.env['VITE_API_URL']);

	export let config: Interface;

	$: store_trigger = async (store_id: string, value: any) => {
		let modifications = await modify(`${$api_url}/store/${store_id}/trigger`, { value: value });
		return modifications;
	};

	$: get_schema = async (box_id: string, columns: Array<string> | null = null) => {
		return await post(`${$api_url}/df/${box_id}/schema`, { columns: columns });
	};

	$: get_rows = async (
		box_id: string,
		start?: number,
		end?: number,
		indices?: Array<number>,
		columns?: Array<string>,
		key_column?: string,
		keys?: Array<string | number>,
	) => {
		let result = await post(`${$api_url}/df/${box_id}/rows`, {
			start: start,
			end: end,
			indices: indices,
			key_column: key_column,
			keys: keys, 
			columns: columns
		});
		return result;
	};

	$: add = async (box_id: string, column_name: string) => {
		let modifications = await modify(`${$api_url}/df/${box_id}/add`, { column: column_name });
		return modifications;
	};

	$: edit = async (
		box_id: string,
		value: string | number,
		column: string,
		row_id: string | number,
		id_column: string
	) => {
		let modifications = await modify(`${$api_url}/df/${box_id}/edit`, {
			value: value,
			column: column,
			row_id: row_id,
			id_column: id_column
		});
		return modifications;
	};

	$: edit_target = async (
		box_id: string,
		target: EditTarget,
		value: any,
		column: string,
		row_indices: Array<number>,
		row_keys: Array<string>,
		primary_key: string,
		metadata: any
	) => {
		let modifications = await modify(`${$api_url}/df/${box_id}/edit_target`, {
			target: target,
			value: value,
			column: column,
			row_indices: row_indices,
			row_keys: row_keys,
			primary_key: primary_key,
			metadata: metadata
		});
		return modifications;
	};

	$: match = async (box_id: string, input: string, query: string, col_out: Writable<string>) => {
		let modifications = await modify(`${$api_url}/ops/${box_id}/match`, {
			input: input,
			query: query,
			col_out: col_out.store_id
		});
		return modifications;
	};

	$: get_sliceby_info = async (box_id: string) => {
		return await get_request(`${$api_url}/sliceby/${box_id}/info`);
	};

	$: get_sliceby_rows = async (
		box_id: string,
		slice_key: SliceKey,
		start?: number,
		end?: number
	) => {
		return await post(`${$api_url}/sliceby/${box_id}/rows`, {
			slice_key: slice_key,
			start: start,
			end: end
		});
	};

	$: aggregate_sliceby = async (box_id: string, aggregations: { string: { id: string } }) => {
		let out = Object();
		for (const [name, aggregation] of Object.entries(aggregations)) {
			out[name] = await post(`${$api_url}/sliceby/${box_id}/aggregate/`, {
				aggregation_id: aggregation.id,
				accepts_df: true
			});
		}
		return out;
	};

	$: remove_row_by_index = async (box_id: string, row_index: number) => {
		let modifications = await modify(`${$api_url}/df/${box_id}/remove_row_by_index`, {
			row_index: row_index
		});
		return modifications;
	};

	const _get_schema = writable(get_schema);
	const _add = writable(add);
	const _match = writable(match);
	const _get_rows = writable(get_rows);
	const _edit = writable(edit);
	const _edit_target = writable(edit_target);
	const _store_trigger = writable(store_trigger);
	const _get_sliceby_info = writable(get_sliceby_info);
	const _aggregate_sliceby = writable(aggregate_sliceby);
	const _get_sliceby_rows = writable(get_sliceby_rows);
	const _remove_row_by_index = writable(remove_row_by_index);

	$: $_get_schema = get_schema;
	$: $_add = add;
	$: $_match = match;
	$: $_get_rows = get_rows;
	$: $_edit = edit;
	$: $_edit_target = edit_target;
	$: $_store_trigger = store_trigger;
	$: $_get_sliceby_info = get_sliceby_info;
	$: $_aggregate_sliceby = aggregate_sliceby;
	$: $_get_sliceby_rows = get_sliceby_rows;
	$: $_remove_row_by_index = remove_row_by_index;

	$: context = {
		get_schema: _get_schema,
		add: _add,
		match: _match,
		get_rows: _get_rows,
		edit: _edit,
		edit_target: _edit_target,
		store_trigger: _store_trigger,
		get_sliceby_info: _get_sliceby_info,
		aggregate_sliceby: _aggregate_sliceby,
		get_sliceby_rows: _get_sliceby_rows,
		remove_row_by_index: _remove_row_by_index
	};
	$: setContext('Interface', context);

	// check if config.components is an array
	let component_array: Array<any>;
	if (Array.isArray(config.components)) {
		component_array = config.components;
	} else {
		component_array = Object.entries(config.components).map(([key, value]) => value);
	}

	let imported_layout: any;
	onMount(async () => {
		imported_layout = (await import(`$lib/layouts/${config.layout.name}.svelte`)).default;
		document.title = config.name;
	});

	for (let i = 0; i < component_array.length; i++) {
		let component = component_array[i];

		// Define the stores
		component.props = nestedMap(component.props, (v: any) => {
			if (!v) {
				return v;
			}
			if (v.store_id !== undefined) {
				// unpack the store
				if (!global_stores.has(v.store_id)) {
					// add it to the global_stores Map if it isn't already there
					let store = meerkat_writable(v.value);
					store.store_id = v.store_id;
					store.backend_store = v.has_children;
					global_stores.set(v.store_id, store);
				}
				return global_stores.get(v.store_id);
			} else if (v.box_id !== undefined) {
				if (!global_stores.has(v.box_id)) {
					// add it to the global_stores Map if it isn't already there
					global_stores.set(v.box_id, writable(v));
				}
				return global_stores.get(v.box_id);
			}
			return v;
		});

		// for (let [k, v] of Object.entries(component.props)) {
		// 	if (v) {
		// 		if (v.store_id !== undefined) {
		// 			// unpack the store
		// 			if (!global_stores.has(v.store_id)) {
		// 				// add it to the global_stores Map if it isn't already there
		// 				let store = meerkat_writable(v.value);
		// 				store.store_id = v.store_id;
		// 				store.backend_store = v.has_children;
		// 				global_stores.set(v.store_id, store);
		// 			}
		// 			component.props[k] = global_stores.get(v.store_id);
		// 		} else if (v.box_id !== undefined) {
		// 			if (!global_stores.has(v.box_id)) {
		// 				// add it to the global_stores Map if it isn't already there
		// 				global_stores.set(v.box_id, writable(v));
		// 			}
		// 			component.props[k] = global_stores.get(v.box_id);
		// 		}
		// 	}
		// }

		// Setup for responsive grid layout
		//		let grid_items = [];
		// grid_items.push({
		// 	6: gridHelp.item({
		// 		x: 0,
		// 		y: 2 * i,
		// 		w: 6,
		// 		h: 2,
		// 		customDragger: true
		// 	}),
		// 	id: i
		// });
	}

	const cols = [[1200, 6]];
</script>

<!-- <div class="w-full">
	<Grid
		bind:items={grid_items}
		rowHeight={50}
		let:index
		let:item
		let:dataItem
		let:movePointerDown
		{cols}
		fastStart={true}
		fillSpace={true}
		scroller={document_container}
	>
		{@const { component, component_id, props } = config.components[index]}
		{@const Component = imported_components[component]}
		<svelte:component this={Component} {...props} />
		{#if item.customDragger}
			<div class="dragger" on:pointerdown={movePointerDown}>
				<Draggable size={20} />
			</div>
		{/if}
	</Grid>
</div> -->

<!-- TODO: Things that are not in the computation graph should have a blank callback. -->

<div class="h-screen">
	{#each Array.from(global_stores.keys()) as store_id}
		<StoreComponent
			{store_id}
			store={global_stores.get(store_id)}
			is_backend_store={global_stores.get(store_id).backend_store}
		/>
	{/each}
	<svelte:component
		this={imported_layout}
		components={config.components}
		{...config.layout.props}
	/>
</div>

<style>
	.dragger {
		@apply opacity-0 hover:opacity-100 absolute top-0 left-0 select-none cursor-grab bg-violet-200 text-violet-600;
	}
</style>

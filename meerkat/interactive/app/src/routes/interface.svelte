<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		const id = url.searchParams.get('id');
		const response = await fetch(`${get(api_url)}/interface/${id}/config`);

		return {
			status: response.status,
			props: {
				config: response.ok && (await response.json())
			}
		};
	}
</script>

<script lang="ts">
	import { get, writable, type Writable } from 'svelte/store';
	import { api_url } from './network/stores';
	import { setContext } from 'svelte';

	import { global_stores, backend_updated_stores } from '$lib/components/blanks/stores';
	import { MatchCriterion } from '$lib/api/datapanel';
	import StoreComponent from '$lib/component/StoreComponent.svelte';
 
	import { onMount } from 'svelte';
	import { modify, post } from '$lib/utils/requests';

	import Grid from 'svelte-grid';
	import gridHelp from 'svelte-grid/build/helper/index';
	import Draggable from 'carbon-icons-svelte/lib/Draggable.svelte';

	import {
		PriorityQueue,
		MinPriorityQueue,
		MaxPriorityQueue,
		type ICompare,
		type IGetCompareValue,
	} from '@datastructures-js/priority-queue';

	export let config: any;

	$: store_trigger = async (store_id: string, value: any) => {
		// TODO Make this API
		let modifications = await post(`${$api_url}/stores/${store_id}/trigger`, { value: value });
		// run on_backend_result here
		// then return
	};

	$: on_backend_result = async (store_modifications: Array<any>) => {
		// When the backend returns StoreModifications
		// put the set of store_ids being updated into the 
		for (let {store_id, store_value} of store_modifications) {
			// Update the store
			// We assume this block runs atomically
				// The problem here could be that another part of the frontend updates
				// store_id's store between these 2 statements
			$backend_updated_stores.add(store_id);
			global_stores.get(store_id).set(store_value);
		}

		// // Clear all the excess stores and temporary store values
		// for (let store_id of $excess_stores) {
		// 	store_trigger(store_id, $temp.get(store_id).value);
		// }
		// $excess_stores.clear();
		// $temp.clear();
	};

	$: get_schema = async (box_id: string, columns: Array<string> | null = null) => {
		return await post(`${$api_url}/dp/${box_id}/schema`, { columns: columns });
	};

	$: get_rows = async (
		box_id: string,
		start?: number,
		end?: number,
		indices?: Array<number>,
		columns?: Array<string>
	) => {
		return await post(`${$api_url}/dp/${box_id}/rows`, {
			start: start,
			end: end,
			indices: indices,
			columns: columns
		});
	};

	$: add = async (box_id: string, column_name: string) => {
		let modifications = await modify(`${$api_url}/dp/${box_id}/add`, { column: column_name });
		return modifications;
	};

	$: match = async (
        box_id: string, 
        input: string, 
        query: string,
        col_out: Writable<string>
    ) => {
		let modifications = await modify(`${$api_url}/ops/${box_id}/match`, {
			input: input,
			query: query,
            col_out: col_out.store_id
		});
		return modifications;
	};

	$: update_stores = async (
		stores: Object, 
    ) => {
		let store_ids = await modify(`${$api_url}/update_stores`, {
			stores: stores,
		});
		return store_ids;
	}

	const _get_schema = writable(get_schema);
	const _add = writable(add);
	const _match = writable(match);
    const _get_rows = writable(get_rows);

	$: $_get_schema = get_schema;
	$: $_add = add;
	$: $_match = match;
    $: $_get_rows = get_rows;

	$: context = {
		get_schema: _get_schema,
		add: _add,
		match: _match,
        get_rows: _get_rows,
	};
	$: setContext('Interface', context);

	let component_refs = new Map();
	for (let { component_id } of config.components) {
		component_refs[component_id] = undefined;
	}

	let document_container;
	let imported_components: any = {};
	onMount(async () => {
		// for loop
		for (let i = 0; i < config.components.length; i++) {
			let component = config.components[i];
			let component_name = component.component;
			imported_components[component_name] = (
				await import(`$lib/component/${component_name.toLowerCase()}/${component_name}.svelte`)
			).default;
		}

		document_container = document.documentElement;
	});

	let grid_items = [];

	for (let i = 0; i < config.components.length; i++) {
		let component = config.components[i];
		// Define the stores
		for (let [k, v] of Object.entries(component.props)) {
			if (v) {
				if (v.store_id !== undefined) {
					// unpack the store
					if (!global_stores.has(v.store_id)) {
						// add it to the global_stores Map if it isn't already there
                        let store = writable(v.value);
                        store.store_id = v.store_id
						global_stores.set(v.store_id, store);
					}
					component.props[k] = global_stores.get(v.store_id);
				} else if (v.box_id !== undefined) {
					if (!global_stores.has(v.box_id)) {
						// add it to the global_stores Map if it isn't already there
						global_stores.set(v.box_id, writable(v));
					}
					component.props[k] = global_stores.get(v.box_id);
				}
			}
		}

		// Setup for responsive grid layout
		grid_items.push({
			6: gridHelp.item({
				x: 0,
				y: 2 * i,
				w: 6,
				h: 2,
				customDragger: true
			}),
			id: i
		});
	}

	interface StoreTrigger {
		store_id: string
		// The source of the trigger - 1=frontend, 0=backend
		source: number
	}

	const compare_store_trigger: ICompare<StoreTrigger> = (a: StoreTrigger, b: StoreTrigger) => {
		return a.source < b.source ? -1 : 1;
	}
	// Queue to keep track of stores that would trigger graph execution on the backend.
	// We order the queue by the timestamp
	const store_trigger_queue = new PriorityQueue<StoreTrigger>(compare_store_trigger);

	// Setup a trigger function that calls the computational graph when stores change
	// TODO (arjundd): Only do this for backend stores.
	let trigger_fn = (store_trigger: StoreTrigger) => {
		// let _store = global_stores.get(store_id);
		// console.log("triggering", $_store);
		// Remove all backend stores from the queue
		let stores_changed_from_frontend = {}
		while (!store_trigger_queue.isEmpty()) {
			const store_trigger = store_trigger_queue.dequeue();
			const store_id = store_trigger.store_id;
			const source = store_trigger.source;
			if (source == 0) { // store was updated from backend
				continue
			} else { // store was updated from frontend
				stores_to_send_to_backend[store_id] = global_stores.get(store_id);
			}
		}

		// Mapping from store_id -> value.
		const store_values = update_stores(stores_changed_from_frontend)

	};
	// $: {
	// 	for (let store_id of global_stores.keys()) {
	// 		store_trigger_queue.push({store_id: store_id, source: 0});
	// 		console.log(store_id);
	// 		// trigger_fn({
	// 		// 	store_id: store_id, 
	// 		// 	timestamp: Date.now()
	// 		// })
	// 	}
	// }

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

{#each Array.from(global_stores.keys()) as store_id}
	<!-- TODO: Things that are not in the computation graph should have a blank callback. -->
	<StoreComponent 
		{store_id}
		store={global_stores.get(store_id)} 
		callback={() => {console.log(store_id, "changed")}} 
	/>
{/each}

<div class="flex-col space-y-3">
	{#each config.components as { component, component_id, props }}
		{@const Component = imported_components[component]}
		<svelte:component this={Component} {...props} />
	{/each}
</div>

<style>
	.dragger {
		@apply opacity-0 hover:opacity-100 absolute top-0 left-0 select-none cursor-grab bg-violet-200 text-violet-600;
	}
</style>

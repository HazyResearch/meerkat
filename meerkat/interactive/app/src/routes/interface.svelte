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

	import { global_stores, meerkat_writable } from '$lib/components/blanks/stores';
	import { MatchCriterion } from '$lib/api/datapanel';
	import StoreComponent from '$lib/component/StoreComponent.svelte';
 
	import { onMount } from 'svelte';
	import { modify, post } from '$lib/utils/requests';

	import Grid from 'svelte-grid';
	import gridHelp from 'svelte-grid/build/helper/index';
	import Draggable from 'carbon-icons-svelte/lib/Draggable.svelte';

	export let config: any;

	$: store_trigger = async (store_id: string, value: any) => {
        console.log("Triggering store", store_id, value);
		let modifications = await modify(`${$api_url}/store/${store_id}/trigger`, { value: value });
		return modifications;
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
		console.log("Inside get_rows");
		console.log(box_id, start, end, indices, columns);
		let result = await post(`${$api_url}/dp/${box_id}/rows`, {
			start: start,
			end: end,
			indices: indices,
			columns: columns
		});
        console.log(result);
        return result 

	};

	$: add = async (box_id: string, column_name: string) => {
		let modifications = await modify(`${$api_url}/dp/${box_id}/add`, { column: column_name });
		return modifications;
	};

    $: edit = async (
        box_id: string, 
        value: string | number,
        column: string,
        row_id: string | number,
        id_column: string,
    ) => {
		let modifications = await modify(
            `${$api_url}/dp/${box_id}/edit`, 
            { 
                value: value, 
                column: column, 
                row_id: row_id, 
                id_column: id_column,
            }
        );
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
    const _edit = writable(edit);
	const _store_trigger = writable(store_trigger);

	$: $_get_schema = get_schema;
	$: $_add = add;
	$: $_match = match;
    $: $_get_rows = get_rows;
    $: $_edit = edit;
	$: $_store_trigger = store_trigger;

	$: context = {
		get_schema: _get_schema,
		add: _add,
		match: _match,
        get_rows: _get_rows,
        edit: _edit,
		store_trigger: _store_trigger,
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
                        let store = meerkat_writable(v.value);
                        store.store_id = v.store_id;
						store.backend_store = v.has_children;
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

<div class="flex flex-col space-y-3 h-screen">

    {#each Array.from(global_stores.keys()) as store_id}
        <!-- TODO: Things that are not in the computation graph should have a blank callback. -->
        <StoreComponent 
            {store_id}
            store={global_stores.get(store_id)} 
			is_backend_store={global_stores.get(store_id).backend_store}
        />
    {/each}

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

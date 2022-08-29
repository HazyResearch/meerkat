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
	import { get, writable } from 'svelte/store';
	import { api_url } from './network/stores';
	import { setContext } from 'svelte';

	import { global_stores } from '$lib/components/blanks/stores';
	import { get_schema, MatchCriterion } from '$lib/api/datapanel';

	import { onMount } from 'svelte';
	import { modify } from '$lib/utils/requests';

	import Grid from "svelte-grid";
	import gridHelp from "svelte-grid/build/helper/index";
	import Draggable from "carbon-icons-svelte/lib/Draggable.svelte";


	export let config: any;

	$: schema = async (box_id: string) => {
		let schema = await get_schema($api_url, box_id);
		return schema;
	};
	
	$: add = async (box_id: string, column_name: string) => {
		let modifications = await modify(`${$api_url}/dp/${box_id}/add`, { column: column_name });
		return modifications;
	};
	
	$: match = async (box_id: string, criterion: MatchCriterion) => {
		let modifications = await modify(`${$api_url}/dp/${box_id}/match`, {
			input: criterion.column,
			query: criterion.query
		});
		return modifications;
	};

	const _schema = writable(schema);
	const _add = writable(add);
	const _match = writable(match);

	$: $_schema = schema;
	$: $_add = add;
	$: $_match = match;

	$: context = {
		schema: _schema,
		add: _add,
		match: _match
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
						global_stores.set(v.store_id, writable(v.value));
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
			id: i,
		})

	}

	const cols = [
		[ 1200, 6 ],
	];

</script>

<style>
	.dragger {
		@apply opacity-0 hover:opacity-100 absolute top-0 left-0 select-none cursor-grab bg-violet-200 text-violet-600;
	}
</style>

<div class="w-full">
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
		{@const {component, component_id, props} = config.components[index]}
		{@const Component = imported_components[component]}
		<svelte:component this={Component} {...props} />
		{#if item.customDragger}
			<div class="dragger" on:pointerdown={movePointerDown}>
				<Draggable size={20}/>
			</div>
		{/if}
	</Grid>
</div>
  

<!-- <div class="flex-col space-y-3">
	{#each config.components as { component, component_id, props }}
		{@const Component = imported_components[component]}
		<svelte:component this={Component} {...props} />
	{/each}
</div> -->

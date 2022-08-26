<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url, fetch }) {
		console.log("In here.");
		const id = url.searchParams.get('id');
		const response = await fetch(`${get(api_url)}/interface/${id}/config`);

		console.log(get(api_url));
		console.log(response);

		return {
			status: response.status,
			props: {
				config: response.ok && (await response.json())
			}
		};
	}
</script>

<script lang="ts">
	import TableView from '$lib/TableView.svelte';
	import SliceCards from '$lib/components/sliceby/SliceCards.svelte';
	import { get, writable } from 'svelte/store';
	import { api_url } from './network/stores';
	import Prism from '../lib/components/cell/code/Code.svelte';
	import Everything from '$lib/components/blocks/Everything.svelte';

	import { global_stores } from '$lib/components/blanks/stores';

	import { onMount } from 'svelte';


	export let config: any;
	console.log(config);

	let component_refs = new Map();
	for (let {component_id} of config.components) {
		component_refs[component_id] = undefined;
	};
	console.log(component_refs);


		
	let the_store_id;
	let store_values: any = new Map();
	for (let {store_id, value} of config.stores) {
		global_stores[store_id] = writable(value);
		console.log("Store");
		console.log(get(global_stores[store_id]));

		the_store_id = store_id;

		global_stores[store_id].subscribe(
			(value: any) => {
				store_values[store_id] = value;
			}
		);
	}

	$: {
		for (let {store_id} of config.stores) {
			global_stores[store_id].set(store_values[store_id]);
			global_stores[store_id] = global_stores[store_id];
		}
	}

	$: setter = (component_id) => {
		component_refs[component_id].$set({'against': store_values[the_store_id]});
		component_refs[component_id] = component_refs[component_id];
	}
	
	// export function bind(component, name, callback) {
	// 	const index = component.$$.props[name];
	// 	if (index !== undefined) {
	// 		component.$$.bound[index] = callback;
	// 		callback(component.$$.ctx[index]);
	// 	}
	// }

	let imported_components: any = {};
	onMount(async () => {
		// for loop
		for (let i = 0; i < config.components.length; i++) {
			let component = config.components[i].component;
			imported_components[component] = (await import(`$lib/components/blanks/${component}.svelte`)).default;
		}

		// for (let {component_id} of config.components) {
		// 	console.log("Blah");
		// 	if (component_refs[component_id]) {
		// 		console.log(component_refs[component_id].$$);
		// 		// component_refs[component_id].$set({'against': store_values[the_store_id]});
		// 		setter(component_id);
		// 		// component_refs[component_id]['against'] = store_values[the_store_id];
		// 		const index = component_refs[component_id].$$.props['against'];
		// 		// component_refs[component_id].$$.ctx[index] = store_values[the_store_id];
		// 		let callback = (value) => {
		// 			store_values[the_store_id] = value;
		// 		};
		// 		component_refs[component_id].$$.bound[index] = callback;
		// 		// callback(component_refs[component_id].$$.ctx[index]);
				
				
		// 	}
		// };
	});



</script>

<!--   -->
{#each config.components as {component, component_id}}
	{@const Component = imported_components[component]}
	{component_id}
	{#if component === 'Match'}
		<!-- <svelte:component this={Component} bind:this={component_refs[component_id]} bind:against={store_values[the_store_id]} /> -->
		<svelte:component this={Component} bind:this={component_refs[component_id]} against={global_stores[the_store_id]} />
		<!-- <svelte:component this={Component} bind:this={component_refs[component_id]} /> -->
	{:else}
		<svelte:component this={Component} />
	{/if}
{/each}

<div class="bg-slate-100">
	{get(global_stores[the_store_id])}
</div>

<!-- <div class="h-[800px]">
	{#if config.component === 'table'}
		<Everything/>
		<TableView nrows={config.props.nrows} datapanel_id={config.props.dp} />
	{:else if config.component === 'sliceby-cards'}
		<SliceCards {...config.props} />
	{:else}
		<div>Type not recognized.</div>
	{/if}
</div> -->

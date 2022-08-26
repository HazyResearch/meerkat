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
	import { getContext, setContext } from 'svelte';

	import { global_stores } from '$lib/components/blanks/stores';
	import { get_schema } from '$lib/api/datapanel';

	import { onMount } from 'svelte';
	import { post, modify } from '$lib/utils/requests';

	export let config: any;

	$: schema = async (box_id: string) => {
		let schema = await get_schema($api_url, box_id);
		return schema;
	};
	const _schema = writable(schema);
	$: $_schema = schema;

	$: add = async (box_id: string, column_name: string) => {
		let modifications = await modify(`${$api_url}/dp/${box_id}/add`, { column: column_name });
		return modifications;
	};
	const _add = writable(add);
	$: $_add = add;

	$: context = {
		schema: _schema,
		add: _add
	};
	$: setContext('Interface', context);

	let component_refs = new Map();
	for (let { component_id } of config.components) {
		component_refs[component_id] = undefined;
	}

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
	});

	for (let i = 0; i < config.components.length; i++) {
		let component = config.components[i];
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
	}
</script>

<div class="flex-col space-y-3">
	{#each config.components as { component, component_id, props }}
		{@const Component = imported_components[component]}
		<svelte:component this={Component} {...props} />
	{/each}
</div>

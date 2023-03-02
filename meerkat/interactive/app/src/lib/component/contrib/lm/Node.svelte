<script lang="ts">
	// import { get_lm_categorization } from '$lib/api/llm';
	import { Draggable, MagicWand, Subtract } from 'carbon-icons-svelte';
	import { getContext, setContext } from 'svelte';
	import { dndzone, SHADOW_PLACEHOLDER_ITEM_ID } from 'svelte-dnd-action';
	import { flip } from 'svelte/animate';
	import { writable } from 'svelte/store';
	import { api_url } from '../../../routes/network/stores';

	const expand_all = getContext('expand_all');
	const hide_add_nodes = getContext('hide_add_nodes');
	const parent_selected = getContext('parent_selected');
	// Set of selected children of this node's parent
	const selected_children = getContext('selected_children');

	const { get_lm_categorization } = getContext('Meerkat');

	const flipDurationMs = 50;

	export let expanded = false;
	export let selected = writable(false);
	setContext('parent_selected', selected);
	export let nodes: any = undefined;
	export let node: any;
	export let add_node = false;

	// Any children of this node that are selected
	const my_selected_children = writable(new Set());
	setContext('selected_children', my_selected_children);

	$: {
		$selected = $parent_selected;
	}

	$: {
		if ($selected) {
			$selected_children.add(node.id);
		} else {
			$selected_children.delete(node.id);
		}
		node.selected = $selected;
	}

	$: {
		expanded = $expand_all;
	}

	function toggle() {
		if (node.items.length > 0 || !$hide_add_nodes) {
			expanded = !expanded;
		}
	}

	function handleDndConsider(e) {
		// Shadow element is in a new position
		node.items = e.detail.items;
	}
	function handleDndFinalize(e) {
		// Shadow element is finalized
		node.items = e.detail.items;
		nodes = { ...nodes };
	}

	let node_input;
	let new_category: string = '';

	let add_category = (e) => {
		if (e.key === 'Enter') {
			e.preventDefault();
			// Add a new node
			let new_node = {
				name: new_category,
				id: `node${nodes.next_id}`,
				items: []
			};
			nodes.next_id++;
			// Let the parent know that a new node has been added
			nodes[node.id.replace('-adder', '')].items.push({ id: new_node.id });
			nodes = { ...nodes, [new_node.id]: new_node };
			new_category = '';
		}
	};

	let add_category_magic = async (e) => {
		let this_node_id = node.id.replace('-adder', '');
		let category_generation_response = await get_lm_categorization(
			nodes[this_node_id].name,
			nodes[this_node_id].items.map((item) => nodes[item.id].name)
		);
		// let category_generation_response = await get_lm_categorization(
		// 	$api_url,
		// 	nodes[this_node_id].name,
		// 	nodes[this_node_id].items.map((item) => nodes[item.id].name)
		// );
		for (let category of category_generation_response.categories) {
			let new_node = {
				name: category,
				id: `node${nodes.next_id}`,
				items: []
			};
			nodes.next_id++;
			// Let the parent know that a new node has been added
			nodes[this_node_id].items.push({ id: new_node.id });
			nodes = { ...nodes, [new_node.id]: new_node };
		}
	};
</script>

<div class="flex items-center justify-start">
	{#if !add_node}
		<div
			class="hover:text-red-500 {$selected
				? 'text-red-500 bg-red-200'
				: 'text-red-300 bg-violet-500'} text-center cursor-pointer font-mono rounded-full mx-1 w-6 h-6"
			on:click={() => {
				$selected = !$selected;
			}}
		>
			<Subtract size="24" />
		</div>
	{/if}
	<div
		class:expanded
		class="cursor-pointer text-violet-600 font-mono"
		on:click={toggle}
		on:keyup={(e) => {
			if (e.key === 'ArrowRight') {
				toggle();
			}
		}}
	>
		{#if add_node}
			<div class="flex h-fit">
				<input
					type="text"
					placeholder="enter new node"
					bind:value={new_category}
					class="p-1 mr-0 ml-0 m-1 text-violet-800 bg-slate-200 text-xs text-ellipsis font-bold whitespace-nowrap select-none rounded-md"
					on:keyup={add_category}
				/>
				<div
					class="w-6 h-6 m-1 flex flex-col justify-center items-center self-center hover:text-red-500 text-center cursor-pointer text-violet-600 bg-violet-300 rounded"
					on:click={add_category_magic}
				>
					<MagicWand size="20" />
				</div>
			</div>
		{:else}
			<div class="flex">
				<input
					type="text"
					bind:value={node.name}
					class="my-1 rounded-l-md text-violet-800 bg-slate-200 text-xs text-ellipsis border-none focus:outline-none"
					bind:this={node_input}
					on:click|stopPropagation
					on:keyup|stopPropagation
					on:focus={(e) => {
						e.target.selectionStart = node_input.value.length;
						e.target.selectionEnd = node_input.value.length;
					}}
				/>
				<div
					class="flex items-center pl-2 pr-2 my-1 rounded-r-md bg-slate-200 text-xs text-ellipsis select-none font-mono"
				>
					{node.items.length ? `(${node.items.length})` : ``}
					<div class="ml-2 pl-2 border-l-2 border-violet-400">
						<Draggable size="20" />
					</div>
				</div>
			</div>
		{/if}
	</div>
</div>
{#if expanded && !add_node}
	{#if !$hide_add_nodes}
		<div class="pl-16 ml-4 border-l border-solid overflow-y-auto">
			<svelte:self
				node={{ id: `${node.id}-adder`, name: 'Adder', items: [] }}
				bind:nodes
				add_node={true}
			/>
		</div>
	{/if}
	{#if node.hasOwnProperty('items')}
		<section
			use:dndzone={{ items: node.items, flipDurationMs }}
			on:consider={handleDndConsider}
			on:finalize={handleDndFinalize}
			class="pl-8 pb-2 ml-4 border-l border-solid overflow-y-auto"
			id="{node.id}-section"
		>
			<!-- WE FILTER THE SHADOW PLACEHOLDER THAT WAS ADDED IN VERSION 0.7.4, filtering this way rather than checking whether 'nodes' have the id became possible in version 0.9.1 -->
			{#each node.items.filter((item) => item.id !== SHADOW_PLACEHOLDER_ITEM_ID) as item (item.id)}
				<div animate:flip={{ duration: flipDurationMs }}>
					<svelte:self bind:nodes node={nodes[item.id]} />
				</div>
			{/each}
		</section>
	{/if}
{/if}

<style>
	.expanded {
		@apply font-bold;
	}
</style>

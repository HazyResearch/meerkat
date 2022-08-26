<script lang="ts">
    import { setContext } from 'svelte';
import { writable } from 'svelte/store';

	import Node from './Node.svelte';

    const expand_all = writable(false);
    const hide_add_nodes = writable(false);
    setContext('expand_all', expand_all);
    setContext('hide_add_nodes', hide_add_nodes);
    setContext('parent_selected', writable(false));
    setContext('selected_children', writable(new Set()));

	let delete_all_children = (node_id, parent_selected = false) => {
        for (let child of nodes[node_id].items) {
            delete_all_children(child.id, nodes[node_id].selected || parent_selected);
        }
		if (node_id !== 'node1' && (nodes[node_id].selected || parent_selected)) {
			delete nodes[node_id];
		} else {
			// Update nodes[node_id].items
			nodes[node_id].items = nodes[node_id].items.filter(item => nodes.hasOwnProperty(item.id));
		}
		
    };


    let root;

	let nodes = {
		node1: {
			name: 'schema',
			items: [],
			id: 'node1',
			selected: false
		}
	};

	// let nodes = {
	// 	node1: {
	// 		name: 'node 1',
	// 		items: [{ id: 'node2' }, { id: 'node3' }, { id: 'node4' }],
	// 		id: 'node1',
	// 		selected: false
	// 	},
	// 	node2: {
	// 		name: 'node 2',
	// 		items: [{ id: 'node5' }, { id: 'node6' }, { id: 'node7' }, { id: 'node8' }],
	// 		id: 'node2',
	// 		selected: false
	// 	},
	// 	node3: {
	// 		name: 'node 3',
	// 		items: [{ id: 'node9' }, { id: 'node10' }, { id: 'node11' }, { id: 'node12' }],
	// 		id: 'node3',
	// 		selected: false
	// 	},
	// 	node4: {
	// 		name: 'node 4',
	// 		items: [{ id: 'node13' }, { id: 'node14' }, { id: 'node15' }, { id: 'node16' }],
	// 		id: 'node4',
	// 		selected: false
	// 	}
	// };
	// // Leaves
	// for (let i = 5; i < 17; i++) {
	// 	nodes[`node${i}`] = { id: `node${i}`, name: `node ${i}`, items: [], selected: false };
	// }
	// Property to keep track of IDs
	nodes['next_id'] = Array.from(Object.keys(nodes)).length + 1;
</script>


<button on:click={() => $expand_all = !$expand_all} class="bg-violet-200 rounded-sm p-1 font-mono text-sm text-violet-600">Expand All {$expand_all}</button>
<button on:click={() => $hide_add_nodes = !$hide_add_nodes} class="bg-violet-200 rounded-sm p-1 font-mono text-sm text-violet-600">Hide Add Nodes {$hide_add_nodes}</button>
<button on:click={() => delete_all_children('node1')} class="bg-violet-200 rounded-sm p-1 font-mono text-sm text-violet-600">Delete Selection</button>
<button on:click={() => console.log(nodes)} class="bg-violet-200 rounded-sm p-1 font-mono text-sm text-violet-600">View Nodes</button>
<Node node={nodes.node1} bind:nodes={nodes} expanded={true} add_node={false} bind:this={root}/>

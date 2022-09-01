<script lang="ts">
import { filter } from 'underscore';

	import Block from './Block.svelte';
	import Dummy from './Dummy.svelte';
	import DummyPlot from './DummyPlot.svelte';
	import DummyTable from './DummyTable.svelte';
	import { _backend, _data } from './stores.js';

	// $: backend = new Map(Object.entries({
	// 	dp_1: [[11, 21, 31], [21, 31, 41]],
	// 	dp_2: [[12, 22, 32], [22, 32, 42]],
	// 	dp_3: [[13, 23, 33], [23, 33, 43]],
	// 	dp_4: [[14, 24, 34], [24, 34, 44]],
	// }));
	// $: $_backend = new Map([ ...$_backend, ...backend ]);

	// let index_backend = (dp_id: string, cols: Array<number>, rows: Array<number>) => {
	// 	return $_backend[dp_id]
	// 		.filter((col, idx) => cols.includes(idx))
	// 		.map(element => element.filter((row, idx) => rows.includes(idx)));
	// }


	// $: grab = (dp_id: string, cols: Array<number>, rows: Array<number>) => {
	// 	// Check if dp_id is in `data`
	// 	if ($_data.has(dp_id)) {
	// 		if $_data[dp_id].has([cols, rows]) {
				
	// 		}
	// 		return index_backend(dp_id, cols, rows);
	// 	}
	// }

	// // Central store of data
	// $: data = {
	// 	dp_1: {
	// 		// These correspond to the parts of dp_1 that are active on the frontend
	// 		// This is a centralized store, so it doesn't say which blocks are
	// 		// using which parts of the data.
	// 		// TODO: when a get_rows action is dispatched, the action should first
	// 		// check if it can be satisfied by the data in the store, and if not,
	// 		// dispatch a get_rows action to the backend.
	// 		q1: [11, 21, 31],
	// 		q2: [11, 21, 31, 41]
	// 	},
	// 	dp_2: {
	// 		q1: [12, 22, 32],
	// 		q2: [12, 22, 32, 42]
	// 	},
	// 	dp_3: {
	// 		q1: [13, 23, 33],
	// 		q2: [13, 23, 33, 43]
	// 	},
	// 	dp_4: {
	// 		q1: [14, 24, 34],
	// 		q2: [14, 24, 34, 44]
	// 	}
	// };
	// $: data = new Map()
	// $: $_data = new Map([ ...$_data, ...data ]);

	// TODO: how to build this object below for arbitrary
	// components
	// This dummy component uses 4 datapanels
	let datapanels = {
		// dp_1 can be modified by the Block's components by calling dispatch()
		// if modified by this Block or any other, dp_1 will be replaced with the modified dp_id
		dp: { id: 'dp_1', isolated: false, view_only: false },
		// dp_1 can be modified by the Block's components by calling dispatch()
		// if modified by this Block or any other, dp_1 will be replaced with the modified dp_id
		dp_a: { id: 'dp_1', isolated: false, view_only: false },
		// dp_2 cannot be modified by the Block's components
		// if dp_2 is modified by another Block, it will be replaced with the modified dp_id
		dp_b: { id: 'dp_2', isolated: false, view_only: true },
		// dp_3 can be modified by the Block's components by calling dispatch()
		// if modified by this Block, dp_3 will be replaced with the modified dp_id
		// if modified by another Block, dp_3 will be unaffected
		dp_c: { id: 'dp_3', isolated: true, view_only: false },
		// dp_4 cannot be modified by this or any other Block's components
		dp_d: { id: 'dp_4', isolated: true, view_only: true }
	};
	// `data` is an object of prior get_rows queries that are still in memory.
	// perhaps data should be in a store that can be looked up by id?

	let my_custom_operation = () => {
		console.log('my_custom_operation');
		$_data[datapanels.dp_c.id].q1.push(9);
		datapanels = datapanels;
		console.log(datapanels);
		console.log('done');
	};

	let count = 0;
	let active: any = {};
</script>

<!-- Scenario 1 -->
<!-- <button class="bg-red-400" on:click={my_custom_operation}>big red button</button>
<Block bind:datapanels>
    <Dummy/>
    <DummyTable/>
</Block>
<Block bind:datapanels>
    <Dummy/>
</Block> -->



<!-- Scenario 2 -->
<Block bind:datapanels>
	<!-- Block containing a view-only Table with pagination -->
	<Block
		datapanels={{
			dp: { id: 'dp_1', isolated: false, view_only: false }
		}}
		bind:active
	>
	<!-- count is a placeholder for the `active` part of the datapanel -->
		{#if active}
			{Object.entries(active)}
		{/if}
		<DummyTable bind:count />
	</Block>
	<!-- Represents a Block containing a view-only Gallery with pagination -->
	<Block
		datapanels={{
			dp: { id: 'dp_1', isolated: false, view_only: true }
		}}
	>
		<DummyTable bind:count />
	</Block>
	<!-- Block containing a view-only Plot -->
	<Block
		datapanels={{
			dp: { id: 'dp_1', isolated: false, view_only: true }
		}}
	>
		<DummyPlot bind:count />
	</Block>
</Block>

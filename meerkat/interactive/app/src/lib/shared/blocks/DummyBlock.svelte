<script lang="ts">
import { filter } from 'underscore';

	import Block from './Block.svelte';
	import Dummy from './Dummy.svelte';
	import DummyPlot from './DummyPlot.svelte';
	import DummyTable from './DummyTable.svelte';
	import { _backend, _data } from './stores.js';

	// $: backend = new Map(Object.entries({
	// 	df_1: [[11, 21, 31], [21, 31, 41]],
	// 	df_2: [[12, 22, 32], [22, 32, 42]],
	// 	df_3: [[13, 23, 33], [23, 33, 43]],
	// 	df_4: [[14, 24, 34], [24, 34, 44]],
	// }));
	// $: $_backend = new Map([ ...$_backend, ...backend ]);

	// let index_backend = (df_id: string, cols: Array<number>, rows: Array<number>) => {
	// 	return $_backend[df_id]
	// 		.filter((col, idx) => cols.includes(idx))
	// 		.map(element => element.filter((row, idx) => rows.includes(idx)));
	// }


	// $: grab = (df_id: string, cols: Array<number>, rows: Array<number>) => {
	// 	// Check if df_id is in `data`
	// 	if ($_data.has(df_id)) {
	// 		if $_data[df_id].has([cols, rows]) {
				
	// 		}
	// 		return index_backend(df_id, cols, rows);
	// 	}
	// }

	// // Central store of data
	// $: data = {
	// 	df_1: {
	// 		// These correspond to the parts of df_1 that are active on the frontend
	// 		// This is a centralized store, so it doesn't say which blocks are
	// 		// using which parts of the data.
	// 		// TODO: when a get_rows action is dispatched, the action should first
	// 		// check if it can be satisfied by the data in the store, and if not,
	// 		// dispatch a get_rows action to the backend.
	// 		q1: [11, 21, 31],
	// 		q2: [11, 21, 31, 41]
	// 	},
	// 	df_2: {
	// 		q1: [12, 22, 32],
	// 		q2: [12, 22, 32, 42]
	// 	},
	// 	df_3: {
	// 		q1: [13, 23, 33],
	// 		q2: [13, 23, 33, 43]
	// 	},
	// 	df_4: {
	// 		q1: [14, 24, 34],
	// 		q2: [14, 24, 34, 44]
	// 	}
	// };
	// $: data = new Map()
	// $: $_data = new Map([ ...$_data, ...data ]);

	// TODO: how to build this object below for arbitrary
	// shared
	// This dummy component uses 4 dataframes
	let dataframes = {
		// df_1 can be modified by the Block's shared by calling dispatch()
		// if modified by this Block or any other, df_1 will be replaced with the modified df_id
		df: { id: 'df_1', isolated: false, view_only: false },
		// df_1 can be modified by the Block's shared by calling dispatch()
		// if modified by this Block or any other, df_1 will be replaced with the modified df_id
		df_a: { id: 'df_1', isolated: false, view_only: false },
		// df_2 cannot be modified by the Block's shared
		// if df_2 is modified by another Block, it will be replaced with the modified df_id
		df_b: { id: 'df_2', isolated: false, view_only: true },
		// df_3 can be modified by the Block's shared by calling dispatch()
		// if modified by this Block, df_3 will be replaced with the modified df_id
		// if modified by another Block, df_3 will be unaffected
		df_c: { id: 'df_3', isolated: true, view_only: false },
		// df_4 cannot be modified by this or any other Block's shared
		df_d: { id: 'df_4', isolated: true, view_only: true }
	};
	// `data` is an object of prior get_rows queries that are still in memory.
	// perhaps data should be in a store that can be looked up by id?

	let my_custom_operation = () => {
		console.log('my_custom_operation');
		$_data[dataframes.df_c.id].q1.push(9);
		dataframes = dataframes;
		console.log(dataframes);
		console.log('done');
	};

	let count = 0;
	let active: any = {};
</script>

<!-- Scenario 1 -->
<!-- <button class="bg-red-400" on:click={my_custom_operation}>big red button</button>
<Block bind:dataframes>
    <Dummy/>
    <DummyTable/>
</Block>
<Block bind:dataframes>
    <Dummy/>
</Block> -->



<!-- Scenario 2 -->
<Block bind:dataframes>
	<!-- Block containing a view-only Table with pagination -->
	<Block
		dataframes={{
			df: { id: 'df_1', isolated: false, view_only: false }
		}}
		bind:active
	>
	<!-- count is a placeholder for the `active` part of the dataframe -->
		{#if active}
			{Object.entries(active)}
		{/if}
		<DummyTable bind:count />
	</Block>
	<!-- Represents a Block containing a view-only Gallery with pagination -->
	<Block
		dataframes={{
			df: { id: 'df_1', isolated: false, view_only: true }
		}}
	>
		<DummyTable bind:count />
	</Block>
	<!-- Block containing a view-only Plot -->
	<Block
		dataframes={{
			df: { id: 'df_1', isolated: false, view_only: true }
		}}
	>
		<DummyPlot bind:count />
	</Block>
</Block>

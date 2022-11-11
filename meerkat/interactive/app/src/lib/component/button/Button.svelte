<script lang="ts">
	import { type Writable } from 'svelte/store';
	import { writable } from 'svelte/store';
	import { getContext } from 'svelte';
	import type { Endpoint } from '$lib/utils/types';

	let { dispatch } = getContext('Interface');

	export let title: string = '';
	export let on_click: Endpoint;
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md flex flex-col">
	<button
		on:click={(e) => {
			console.log(e);
			// $on_click = { type: e.type, detail: e.detail };
			let output = $dispatch(on_click.endpoint_id, {}, { type: e.type, detail: e.detail });
			console.log(output);
			output.then((e) => console.log(e));
		}}>{title}</button
	>
</div>

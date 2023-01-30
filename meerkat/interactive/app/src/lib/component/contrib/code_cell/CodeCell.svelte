<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import type { Endpoint } from '$lib/utils/types';

	const { dispatch } = getContext('Meerkat');

	export let on_run: Endpoint;

	let error = null;

	const onKeyPress = (e) => {
		if (e.charCode === 13) run_search();
	};

	let run_search = async (text) => {
		try {
			await $dispatch(
				on_run.endpoint_id,
				{
					new_code: text
				},
				{}
			);
			error = null;
		} catch (err) {
			error = err;
			console.log(err);
		}
	};

	let divEl: HTMLDivElement = null;
	let editor;
	let monaco;
	onMount(async () => {
		// @ts-ignore
		monaco = await import('monaco-editor');

		editor = monaco.editor.create(divEl, {
			value: ['function x() {', '\tconsole.log("Hello world!");', '}'].join('\n'),
			language: 'python'
		});
		editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => {
			run_search(editor.getValue());
		});
		return () => {
			editor.dispose();
		};
	});
</script>

<div bind:this={divEl} class="h-32" />

{#if error}
	<div><h1>{error}</h1></div>
{/if}

<!-- <input
	type="text"
	bind:value={searchValue}
	placeholder="Write some text to be matched..."
	class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
	on:keypress={onKeyPress}
/> -->

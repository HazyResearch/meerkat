<script lang="ts">
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import { onMount } from 'svelte';

	export let query: string;
	export let on_run: Endpoint;



	let run_search = async (text) => {
		await dispatch(on_run.endpointId, {
			detail: {
				new_query: text
			}
		});
	};

	let divEl: HTMLDivElement | null = null;
	let editor;
	let monaco;
	onMount(async () => {
		// @ts-ignore
		monaco = await import('monaco-editor');

		editor = monaco.editor.create(divEl, {
			value: query,
			language: 'python',
			minimap: { enabled: false }
		});
		editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => {
			run_search(editor.getValue());
		});
		return () => {
			editor.dispose();
		};
	});

	$: { 
		if (editor) {
			editor.setValue(query); 
		}
	}

</script>

<div class="h-48 w-full rounded-md border-slate-400 bg-slate-100 pt-2 pb-1 px-1">
	<div bind:this={divEl} class="h-full w-full" />
</div>

<script lang="ts">
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import { onMount } from 'svelte';

    /** The code to display in the editor. */
	export let code: string;

	/** Live update. */
	export let live: boolean = false;

    /** The endpoint to call when the user runs the code. */
	export let on_run: Endpoint;

	let runSearch = async (text) => {
		await dispatch(on_run.endpointId, {
			detail: {
				new_code: text
			}
		});
	};

	let divEl: HTMLDivElement | null = null;
	let editor: any;
	let monaco;
	onMount(async () => {
		// @ts-ignore
		monaco = await import('monaco-editor');

		editor = monaco.editor.create(divEl as HTMLDivElement, {
			value: code,
			language: 'python',
			minimap: { enabled: false }
		});
		editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => {
			runSearch(editor.getValue());
		});

		editor.onDidBlurEditorText(() => {
			code = editor.getValue();
		});

		return () => {
			editor.dispose();
		};
	});

	$: {
		if (editor) {
			editor.setValue(code);
		}
	}
</script>

<div class="h-32 w-full rounded-md border-slate-400 bg-slate-100 pt-2 pb-1 px-1">
	<div 
		id="editor" 
		bind:this={divEl} 
		class="h-full w-full" 
	/>
</div>

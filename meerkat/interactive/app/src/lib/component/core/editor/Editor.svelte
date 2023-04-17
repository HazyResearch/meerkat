<script lang="ts">
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import { onMount } from 'svelte';

    /** The code to display in the editor. */
	export let code: string;
	export let title: string = 'Code';

	/** Live update. */
	export let live: boolean = false;

    /** The endpoint to call when the user runs the code. */
	export let onRun: Endpoint;

	let runSearch = async (text) => {
		await dispatch(onRun.endpointId, {
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
			console.log("heree")
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

<div class="h-full grid grid-rows-[auto_1fr] w-full border-slate-400 pb-1 rounded-sm shadow-md overflow-hidden">
	<div class="w-full pl-4 text-sm text-slate-400">{title}</div>
	<div 
		id="editor" 
		bind:this={divEl} 
		class="w-full -ml-4 -mr-10" 
	/>
</div>


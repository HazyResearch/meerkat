<script lang="ts">
	import { getContext, onMount } from 'svelte';

    import type { Endpoint } from '$lib/utils/types';

    const { dispatch } = getContext('Interface');

	export let on_run: Endpoint;


    const onKeyPress = (e) => {
		if (e.charCode === 13) run_search();
	};

    let run_search = async (text) => {
        console.log("run search");
		$dispatch(on_run.endpoint_id, {
			"new_code": text,
		}, {});
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
<div bind:this={divEl} class="h-64" />

<!-- <input
	type="text"
	bind:value={searchValue}
	placeholder="Write some text to be matched..."
	class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
	on:keypress={onKeyPress}
/> -->
 


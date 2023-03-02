<script lang="ts">
	import { Fileupload } from 'flowbite-svelte';
	import { createEventDispatcher } from 'svelte';

	const eventDispatcher = createEventDispatcher();

	export let files: FileList;
	export let filenames: string[];
	export let contents: string[];
	export let classes: string;

	export let webkitdirectory: boolean = false;
	export let directory: boolean = false;
	export let multiple: boolean = false;

	$: filenames = Array.from(files).map((file) => file.name);

	let getFileListContents = async (files: FileList) => {
		return await Promise.all(Array.from(files).map((file) => file.text()));
	};

	// Dispatch `upload` event with value and the content of each file.text
	let handleUpload = async () => {
		// Get the contents of each file
		contents = await getFileListContents(files);
		eventDispatcher('upload', { filenames: Array.from(files).map((file) => file.name), contents });
	};

	$: files, handleUpload();
</script>

<Fileupload
	bind:files
	inputClass={classes}
	on:change
	on:keyup
	on:keydown
	on:keypress
	on:focus
	on:blur
	on:click
	on:mouseover
	on:mouseenter
	on:mouseleave
	on:paste
	{webkitdirectory}
	{directory}
	{multiple}
/>

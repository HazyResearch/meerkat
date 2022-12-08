<script lang="ts">
	import { type Writable } from 'svelte/store';
	import { onMount } from 'svelte';

	export let df: Writable;
	export let against: Writable<string>;
	export let col: Writable<string>;
	export let text: Writable<string>; //we can get rid of this.
	export let title: string = '';

	let divEl: HTMLDivElement = null;
	let editor;
	let monaco;
	onMount(async () => {
		// @ts-ignore
		monaco = await import('monaco-editor');

		let regex = 'col';
		monaco.languages.register({ id: 'mySpecialLanguage' });

		// Register a tokens provider for the language
		monaco.languages.setMonarchTokensProvider('mySpecialLanguage', {
			tokenizer: {
				root: [
					[/\[error.*/, 'custom-error'],
					[/\[notice.*/, 'custom-notice'],
					[/\[info.*/, 'custom-info'],
					[/\[[a-zA-Z 0-9:]+\]/, 'custom-date']
				]
			}
		});

		// Define a new theme that contains only rules that match this language
		monaco.editor.defineTheme('myCoolTheme', {
			base: 'vs',
			inherit: false,
			rules: [
				{ token: 'custom-info', foreground: '808080' },
				{ token: 'custom-error', foreground: 'ff0000', fontStyle: 'bold' },
				{ token: 'custom-notice', foreground: 'FFA500' },
				{ token: 'custom-date', foreground: '008800' }
			],
			colors: {
				'editor.foreground': '#000000'
			}
		});

		// Register a completion item provider for the new language
		monaco.languages.registerCompletionItemProvider('mySpecialLanguage', {
			provideCompletionItems: () => {
				var suggestions = [
					{
						label: 'simpleText',
						kind: monaco.languages.CompletionItemKind.Text,
						insertText: 'simpleText'
					},
					{
						label: 'testing',
						kind: monaco.languages.CompletionItemKind.Keyword,
						insertText: 'testing(${1:condition})',
						insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
					},
					{
						label: 'ifelse',
						kind: monaco.languages.CompletionItemKind.Snippet,
						insertText: ['if (${1:condition}) {', '\t$0', '} else {', '\t', '}'].join('\n'),
						insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
						documentation: 'If-Else Statement'
					}
				];
				return { suggestions: suggestions };
			}
		});

		editor = monaco.editor.create(divEl, {
			theme: 'myCoolTheme',
			value: ['function x() {', '\tconsole.log("Hello world!");', '}'].join('\n'),
			language: 'mySpecialLanguage'
		});

		return () => {
			editor.dispose();
		};
	});

	// monaco.languages.setMonarchTokensProvider('mySpecialLanguage', {
	// 	tokenizer: {
	// 		root: [
	// 			[/\[error.*/, 'custom-error'],
	// 			[/\[notice.*/, 'custom-notice'],
	// 			[/\[info.*/, 'custom-info'],
	// 			[/\[[a-zA-Z 0-9:]+\]/, 'custom-date']
	// 		]
	// 	}
	// });
</script>

<div bind:this={divEl} class="h-screen" />

<!-- 
<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md z-50 flex flex-col">
	{#if title != ''}
		<div class="font-bold text-xl text-slate-600 self-start pl-2">
			{title}
		</div>
	{/if}
	<div bind:this={divEl} class="h-screen" /> -->
<!-- <div class="form-control">
		<div class="input-group w-100% flex items-center">
			<div class="px-3">
				<Status {status} />
			</div>
			

			<input
				type="text"
				bind:value={searchValue}
				placeholder="Write some text to be matched..."
				class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
				on:keypress={onKeyPress}
			/>
			<div class="text-slate-400 px-2">against</div>

			<div class="themed pr-2 w-48">
				{#await items_promise}
					<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
				{:then items}
					<Select
						id="column"
						placeholder="...a column."
						value={against_item}
						{items}
						showIndicator={true}
						listPlacement="auto"
						on:select={handleSelect}
						on:clear={handleClear}
					/>
				{/await}
			</div>
		</div>
	</div> -->
<!-- </div> -->
<!-- 
<div class="w-full py-5 px-2 bg-slate-100 ">
    Match

    <input type="text" bind:value={$against}>
    <input type="text" bind:value={$text}>


    Column: {$col}

    <button class="bg-slate-500" on:click={on_add}> Add </button>
    
    {#await schema_promise}
        waiting....
    {:then schema}
        <div class="flex space-x-3">
            {#each schema.columns as column_info}  
                <div class="bg-violet-200 rounded-md px-3 font-bold text-slate-700">
                    {column_info.name}
                </div>  
            {/each}
        </div>
    {:catch error}
        {error}
    {/await}

</div> -->

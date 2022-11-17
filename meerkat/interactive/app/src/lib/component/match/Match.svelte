<script lang="ts">
	import { get, writable, type Writable } from 'svelte/store';
	import { MatchCriterion, type DataPanelSchema } from '$lib/api/datapanel';
    import * as monaco from 'monaco-editor';
	import { getContext } from 'svelte';
	import Status from '$lib/components/common/Status.svelte';
	import Select from 'svelte-select';

	import { onMount } from 'svelte';
    import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
    import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
    import cssWorker from 'monaco-editor/esm/vs/language/css/css.worker?worker';
    import htmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker';
    import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';
	import { Model } from 'carbon-icons-svelte';


	const { get_schema, match } = getContext('Interface');

	export let dp: Writable;
	export let against: Writable<string>;
	export let col: Writable<string>;
	export let text: Writable<string>; //we can get rid of this.
	export let title: string = '';


	let status: string = 'waiting';
	let schema_promise;
	let items_promise;
	let searchValue: string="";
	let columns: Array<Object>=[];

	let divEl: HTMLDivElement = null;
    let editor: monaco.editor.IStandaloneCodeEditor;
    let Monaco;


	let keywords: Array<string>=[]; 
	let regex="col\\(";
	//if we wanted to add more regex, it would look like: 
	//i.e. add row as another keyword: 
	//regex = "(col\\()|(row\\()|(tag\\() etc..."
	console.log({regex});

	//onMount will happen after the other calls in the <script/>

	$: {
		schema_promise = $get_schema($dp.box_id);
		items_promise = schema_promise.then((schema: DataPanelSchema) => {
			return schema.columns.filter((column) => {
				return schema.columns.map((col) => col.name).includes(`clip(${column.name})`)
			}).map(column => {
				console.log("this is a column: ");
				console.log({column});
				columns.push(column);
				keywords.push(`col(` + column.name + `)`);
				console.log({columns});
			return({value: column.name, label: column.name})})
		});
	}
	
	onMount(async () => {
		//col prompts
		//img is the value
		monaco.languages.register({id: 'myLang'});
		monaco.languages.setMonarchTokensProvider('myLang', {
			keywords, 
			tokenizer:{
				root:[[
					regex, {
						cases:{
							'@keywords': 'keyword',
							'@default': 'variable',
						} 
					}
				]
				]
			}
		});

		

		monaco.languages.registerCompletionItemProvider('myLang', {
			provideCompletionItems: (model, position) => {
				const suggestions = [
					...keywords.map(keyword => {
						return {
							label: keyword,
							kind: monaco.languages.CompletionItemKind.Keyword,
							insertText: keyword
						}
					})
				]
				return {suggestions: suggestions};
			}
		})
		let options : monaco.editor.IStandaloneEditorConstructionOptions = {
			wordWrap: 'off',
            lineNumbers: 'off',
            lineDecorationsWidth: 0,
            overviewRulerLanes: 0,
            overviewRulerBorder: false,
            scrollbar: { horizontal: 'hidden', vertical: 'hidden' }
			
	};
		monaco.editor.create(divEl,{
			value: "",
            language: 'myLang',
			options:options,
		});
		
    });

	

	
	

	const onKeyPress = (e) => {
		//match with column names. 
		console.log(e.key);
		console.log({columns});
		console.log({searchValue});
		if(searchValue.includes('col(')){
			columns.map(column => {
				console.log({column});
			})
		}
		if (e.charCode === 13) on_search();
		else status = 'waiting';
	};



	let on_search = async () => {
		
		if ($against === '') {
			status = 'error';
			return;
		}
		status = 'working';
		let box_id = $dp.box_id;
		let promise = $match(box_id, $against, $text, col);
		promise
			.then(() => {
				status = 'success';
			})
			.catch((error: TypeError) => {
				status = 'error';
				console.log(error);
			});
	};

	function handleSelect(event) {
		$against = event.detail.value;
	}

	function handleClear() {
		$against = '';
	}
	$: against_item = { value: $against, label: $against };

	
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md z-50 flex flex-col">
	{#if title != ''}
		<div class="font-bold text-xl text-slate-600 self-start pl-2">
			{title}
		</div>
	{/if}
	<div bind:this={divEl} class="h-screen" />
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
</div>
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

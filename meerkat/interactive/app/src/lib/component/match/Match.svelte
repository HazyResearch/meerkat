<script lang="ts">
	import { get, writable, type Writable } from 'svelte/store';
	import { MatchCriterion, type DataPanelSchema } from '$lib/api/datapanel';
    import * as monaco from 'monaco-editor';
	import { getContext, onDestroy } from 'svelte';
	import Status from '$lib/components/common/Status.svelte';
	import Select from 'svelte-select';

	import { onMount } from 'svelte';
    import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
    import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
    import cssWorker from 'monaco-editor/esm/vs/language/css/css.worker?worker';
    import htmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker';
    import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';
	import { Model } from 'carbon-icons-svelte';

	type SuggestionOption = {
		name:string,
		type: string, 
		cell_props: Record<any, any>,
		cell_component: string
	}
	type Suggestion = {
		key: string, 
		name: string, 
	}


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
	

	let divEl: HTMLDivElement = null;
    let editor: monaco.editor.IStandaloneCodeEditor;
    let Monaco;
	
	let columns: SuggestionOption[] = [];
	let columnKeywords: Array<string>=[]; 
	let suggestions: Array<Suggestion>=[{key: 'match(', name:'match'}];
	//if we wanted to add more regex, it would look like: 
	//i.e. add row as another keyword: 
	//regex = "(col\\()|(row\\()|(tag\\() etc..."

	let searchString: string="";
	let matchedSuggestions: Array<Suggestion>=[]; 
	let userClosed: boolean=false;

	let selectionStart: number=0;

	const onBodyClick = () => {
		userClosed = true; 
	}

	document.addEventListener("click", onBodyClick) 
	onDestroy(() => {
		document.body.removeEventListener("click", onBodyClick)
	})



	//onMount will happen after the other calls in the <script/>

	$: {
		schema_promise = $get_schema($dp.box_id);
		items_promise = schema_promise.then((schema: DataPanelSchema) => {
			let clipRegex = /clip\((.*)\)/;
			return schema.columns.filter((column) => 
				column.name.match(clipRegex)
			).map(column => {
				const name = column.name.match(clipRegex)?.[1] ?? column.name;
				suggestions.push({name, key: `col(${name})`});
			return({value: column.name, label: column.name})})
		});
	}


	onMount(async () => {
		//col prompts
		//img is the value
		
		
    });

	const onKeyPress = (e: KeyboardEvent) => {

		//see if search result matches regex
		//if so, filter for all results with that starting
		
		// if(searchString?.match(columnRegex)){
		// 	let results = columns.filter(column => )
		// }

		//match with column names. 
		//console.log(e.key);
		// let results = columns.filter((column) => {
		// 	return column.includes()
		// })

		console.log({columns});
		console.log({searchString});
		// if(searchString.includes('col(')){
		// 	columns.map(column => {
		// 		console.log({column});
		// 		console.log("FOUND");
		// 	})
		// }
		selectionStart = (e.target as HTMLInputElement).selectionStart || 0;
		matchedSuggestions = suggestions.filter((suggestion) => {
			const searchStringTokens = searchString.split(" ");
			const tokenIndex = findCurrentGroupIndex(searchString, selectionStart);
			const token = searchStringTokens[tokenIndex];
			return token && suggestion.key.toLowerCase().match(new RegExp('^' + escapeRegExp(token).toLowerCase()));
		}

		);
		 
		// //if (e.charCode === 13) on_search();
		// else status = 'waiting';
	};

	//see if last character is a space, then cut the word
	//if the user types in col(, automatically add in parentheses and put the clicker in the middle
	//if a user types c, co, col, it should pop up with dropdown
	// colorize text in command line. 

	

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

	const handleChange = (e) => {
		searchValue = e.target.value;
		console.log({searchValue});
		
	}

	const selectSuggestion = (key: string) => (e: MouseEvent) => {
		e.preventDefault();
		const searchStringTokens = searchString.split(" ");
		const tokenIndex = findCurrentGroupIndex(searchString, selectionStart);
		searchStringTokens[tokenIndex] = key;
		const newCursorPosition = searchStringTokens.slice(tokenIndex).join(" ").length;
		searchString = searchStringTokens.join(" ");
		const queryInput = document.getElementById("queryInput") as HTMLInputElement;
		if(queryInput){
			queryInput.focus();
			console.log({newCursorPosition});
			queryInput.setSelectionRange(newCursorPosition, newCursorPosition);
		}
	}

	const findCurrentGroupIndex = (str: string, cursorPosition: number) => {
		const tokens = str.split(" ");
		//cp = 8. 
		let i = 0;
		return tokens.findIndex((token) => {
			i += token.length+1;
			return cursorPosition <= i;
		})
	}

	function escapeRegExp(text: string) {
  		return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&');
	}
	
</script>

<style>
	.command-line{
		background-color: red;
		
	}
</style>


 
<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md z-50 flex flex-col">

	
	{#if title != ''}
		<div class="font-bold text-xl text-slate-600 self-start pl-2">
			{title}
		</div>
	{/if}
	
	<div style="bg-red">
		<input id="queryInput" bind:value="{searchString}" on:keyup={onKeyPress} type="search" name="search" placeholder="Begin your query" class="bg-white h-10 px-5 pr-10 rounded-full text-sm focus:outline-none"/>
	</div>
	<p>This is search string {searchString}</p>

	{#if matchedSuggestions && searchString}
		{console.log("OPEN DROPDOWN")}
		{console.log({userClosed})}
		<div class="bg-white rounded mt-2 text-black overflow-hidden z-50">
			{#each matchedSuggestions as result}
				<a class="search" href="" on:click={selectSuggestion(result.key)}>{result.key}</a>
				<br/>
			{/each}
		</div>
	{/if}
	<!-- <div class="form-control">
		<div class="input-group w-100% flex items-center">
			<div class="px-3">
				<Status {status} />
			</div>
			
			<div bind:this={divEl} style="" class="command-line input mr-5  grow h-10 rounded-md shadow-md" />

		</div>
	</div> -->
</div>

<!-- <input
	type="text"
	bind:value={searchValue}
	placeholder="Write some text to be matched..."
	class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
	on:keypress={onKeyPress}
/> -->
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

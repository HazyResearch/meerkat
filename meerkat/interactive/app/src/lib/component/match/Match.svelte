<script lang="ts">
	import { get, writable, type Writable } from 'svelte/store';
	import { MatchCriterion, type DataFrameSchema } from '$lib/api/dataframe';
	import { getContext, onDestroy } from 'svelte';
	import Status from '$lib/shared/common/Status.svelte';
	import Select from 'svelte-select';
	import type { Endpoint } from '$lib/utils/types';

	const { get_schema, dispatch } = getContext('Interface');

	export let df: Writable;
	export let against: Writable<string>;
	export let on_match: Endpoint;
	export let text: Writable<string>;
	export let title: Writable<string> = '';
	export let get_match_schema: Endpoint;

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
	let columns: SuggestionOption[] = [];
	let suggestions: Suggestion[] = [];
	let matchedSuggestions: Suggestion[] = [];
	let searchString: string="";
	let userClosed: boolean=false;
	let selectionStart: number=0;

	const onBodyClick = () => {
		userClosed = true; 
	}

	document.addEventListener("click", onBodyClick) 
	onDestroy(() => {
		document.body.removeEventListener("click", onBodyClick)
	})

	
	let status: string = 'waiting';

	let schema_promise;
	// let items_promise;
	$: {
		schema_promise = $dispatch(get_match_schema.endpoint_id, {}, {}).then((schema: DataFrameSchema) => {
			console.log(schema.columns);
			return schema.columns.filter((column) => column.name.includes("_clip")).forEach((column)=> {
				const name = column.name.split("_clip")[0];
				suggestions.push({name, key: `col(${name})`});
			})
		});
	}

 	console.log({suggestions});


	console.log(on_match)

	
	const onKeyPress = (e: KeyboardEvent) => {
		console.log({columns});
		console.log({searchString});
		if(searchString.includes('col(')){
			columns.map(column => {
				console.log({column});
				console.log("FOUND");
			})
		}
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




	let on_search = async () => {
		if ($against === '') {
			status = 'error';
			return;
		}
		status = 'working';
		let promise = $dispatch(on_match.endpoint_id, {
			"against": $against,
			"query": $text
		}, {});
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
		console.log("here")
		console.log(against)
		$against = event.detail.value;
	}

	function handleClear() {
		$against = '';
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





	$: against_item = { value: $against, label: $against };
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md z-50 flex flex-col">
	{#if $title != ''}
		<div class="font-bold text-md text-slate-600 pl-2 text-center">
			{$title}
		</div>
	{/if}
	<div class="form-control">
		<div class="input-group w-100% flex items-center">
			<div class="px-3">
				<Status {status} />
			</div>
			<!-- <input
				type="text"
				bind:value={$text}
				placeholder="Write some text to be matched..."
				class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
				on:keypress={onKeyPress}
			/> -->
		
			<div class="relative">
				<input id="queryInput" bind:value="{searchString}" on:keyup={onKeyPress} type="search" name="search" placeholder="Begin your query" class="bg-white h-10 px-5 pr-10 rounded-full text-sm focus:outline-none"/>
		
				{#if matchedSuggestions && searchString}
					{console.log("OPEN DROPDOWN")}
					{console.log({userClosed})}
					<div class="absolute bg-white rounded mt-2 text-black overflow-hidden z-50 top-4 left-0">
						{#each matchedSuggestions as result}
							<a class="search" href="" on:click={selectSuggestion(result.key)}>{result.key}</a>
							<br/>
						{/each}
					</div>
				{/if}
			</div>	
		</div>
	</div>
</div>




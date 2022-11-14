<script lang="ts">
	import { get, writable, type Writable } from 'svelte/store';
	import { MatchCriterion, type DataPanelSchema } from '$lib/api/datapanel';
    import monaco from 'monaco-editor';
	import { getContext } from 'svelte';
	import Status from '$lib/components/common/Status.svelte';
	import Select from 'svelte-select';

	import { onMount } from 'svelte';
    import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
    import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
    import cssWorker from 'monaco-editor/esm/vs/language/css/css.worker?worker';
    import htmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker';
    import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';


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


	onMount(async () => {
        // @ts-ignore
        self.MonacoEnvironment = {
            getWorker: function (_moduleId: any, label: string) {
                if (label === 'json') {
                    return new jsonWorker();
                }
                if (label === 'css' || label === 'scss' || label === 'less') {
                    return new cssWorker();
                }
                if (label === 'html' || label === 'handlebars' || label === 'razor') {
                    return new htmlWorker();
                }
                if (label === 'typescript' || label === 'javascript') {
                    return new tsWorker();
                }
                return new editorWorker();
            }
        };

		let keywords = ['col('];
        Monaco = await import('monaco-editor');
		
		let regex="col\(";
		editor = Monaco.editor.create(divEl, {
            value: ['function x() {', '\tconsole.log("Hello world!");', '}'].join('\n'),
            language: 'mylang'
        });
		
		editor.languages.register({id: 'myLang'});
		editor.languages.setMonarchTokensProvider("myLang", {
			tokenizer:{
				root:[
					[regex, 
				{
					cases:{
						'@keywords': 'keyword',
						'@default': 'variable',
					}
				}],
				]
			}
		});
        

        return () => {
            editor.dispose();
        };
    });

	$: {
		schema_promise = $get_schema($dp.box_id);
		items_promise = schema_promise.then((schema: DataPanelSchema) => {
			return schema.columns.filter((column) => {
				return schema.columns.map((col) => col.name).includes(`clip(${column.name})`)
			}).map(column => {
				console.log("this is a column: ");
				console.log({column});
				columns.push(column);
				console.log(columns);
			return({value: column.name, label: column.name})})
		});
	}
	

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

<script lang="ts">
	import { get, writable, type Writable } from 'svelte/store';
	import { MatchCriterion, type DataPanelSchema } from '$lib/api/datapanel';
	import { getContext } from 'svelte';
	import Status from '$lib/components/common/Status.svelte';
	import Select from 'svelte-select';

	const { schema, add, match } = getContext('Interface');

	export let dp: Writable;
	export let against: Writable<string>;
	export let col: Writable<string>;
	export let text: Writable<string>;

	let status: string = 'waiting';

    let schema_promise;
    let items_promise;
	$: {
		schema_promise = $schema($dp.box_id);
		items_promise = schema_promise.then((schema: DataPanelSchema) => {
			return schema.columns.map((column) => {
				return {
					value: column.name,
					label: column.name
				};
			});
		});
	}

	let on_add = async () => {
		let box_id = $dp.box_id;
		let column_name = `add(${$against})`;
		col.set(column_name);
		$add(box_id, column_name);
	};

	const onKeyPress = (e) => {
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
		let match_criterion = new MatchCriterion($against, $text);
        let promise = $match(box_id, match_criterion);
		promise.then(() => {
			status = 'success';
		});
	};

	function handleSelect(event) {
		$against = event.detail.value;
	}

	function handleClear() {
		$against = '';
	}
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md">
	<div class="form-control w-full">
		<div class="input-group w-100% flex items-center">
			<div class="px-3">
				<Status {status} />
			</div>
			<input
				type="text"
				bind:value={$text}
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
						{items}
						showIndicator={true}
						listPlacement="auto"
						on:select={handleSelect}
						on:clear={handleClear}
					/>
				{/await}
			</div>
		</div>
	</div>
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

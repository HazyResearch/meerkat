<script lang="ts">
	import { get_categorization } from '$lib/shared/deprecate/lm/llm';
	import { field, form } from 'svelte-forms';
	import { required } from 'svelte-forms/validators';
	import { writable } from 'svelte/store';
	import { api_url } from '../../../routes/network/stores';
	import SchemaTree from './SchemaTree.svelte';

	export let categories: Array<string> = [];
	const dataset_description = field('dataset_description', undefined, [required()], {
		checkOnInit: true
	});
	const hint = field('hint', undefined, [required()], { checkOnInit: true });
	const category_generator_form = form(dataset_description, hint);

	export let cols: Array<string> = [];
	const stored_categorization = writable(new Map());
	$: {
		if ($stored_categorization.has($dataset_description.value)) {
			categories = $stored_categorization.get($dataset_description.value);
		} else {
			categories = [];
		}
	}

	$: {
		if (Array.from($stored_categorization.keys()).length > 0) {
			cols = Array.from($stored_categorization.keys());
		} else {
			cols = [];
		}
	}

	let on_keypress = async (e) => {
		if (e.key === 'Enter') {
			e.preventDefault();
			submit_form(e);
		} else {
			if ($stored_categorization.has($dataset_description.value)) {
				categories = $stored_categorization.get($dataset_description.value);
			}
		}
	};

	let submit_form = async (e) => {
		if (!$stored_categorization.has($dataset_description.value)) {
			$stored_categorization.set($dataset_description.value, []);
		}
		if ($dataset_description.valid) {
			// let category_generation_response = await get_categories($api_url, $dataset_description.value, $hint.value);
			console.log('submit_form');
			let category_generation_response = await get_categorization(
				$api_url,
				$dataset_description.value,
				categories
			);
			console.log(category_generation_response);
			$stored_categorization
				.get($dataset_description.value)
				.push(...category_generation_response.categories);
			$stored_categorization = $stored_categorization;
		}
	};

	let add_category = (e) => {
		if (e.key === 'Enter') {
			e.preventDefault();
			if ($stored_categorization.has($dataset_description.value)) {
				$stored_categorization.get($dataset_description.value).push($hint.value);
			} else {
				$stored_categorization.set($dataset_description.value, [$hint.value]);
			}
			categories = $stored_categorization.get($dataset_description.value);
			$stored_categorization = $stored_categorization;
			$hint.value = '';
		}
	};
</script>

<SchemaTree />
<!-- <div class="w-full h-full text-violet-400">
    <input
        type="text"
        class="input input-bordered grow w-full h-10 rounded-md shadow-md"        
        placeholder="Column..."
        bind:value={$dataset_description.value}
        on:keypress={on_keypress}
    />
    <div class="flex flex-wrap">
        {#each cols as col}
            <div class="ml-1">
                <Pill layout="wide-header" header={col}/>
            </div>
        {/each}
    </div>
    <div class="flex flex-wrap">
        <input
            type="text"
            class="input input-bordered basis-1/4 h-10 mt-1 rounded-md shadow-md {(!$dataset_description.value) ? `disabled:opacity-75` : ``}" 
            class:cursor-not-allowed={!$dataset_description.value}
            placeholder="Add category..."
            disabled={!$dataset_description.value}
            readonly={!$dataset_description.value}
            bind:value={$hint.value}
            on:keyup={add_category}
        />
        <button 
            class="ml-4 rounded border pl-1 pr-1 mt-1 shadow-md font-light text-lg bg-violet-200"
            on:click={submit_form}
        >
            <div class="border border-solid pl-1 pr-1 mt-[2px] mb-[2px] bg-violet-50">LM</div>
        </button>
    </div>
    <div class="flex flex-wrap">
        {#each categories as category}
            <div class="ml-1">
                <Pill layout="wide-header" header={category}/>
            </div>
        {/each}
    </div>
</div> -->

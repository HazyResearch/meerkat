<script lang="ts">
	import { form, field } from 'svelte-forms';
	import { required } from 'svelte-forms/validators';
	import { get_lm_categorization } from '$lib/api/llm';
	import { api_url } from '../../../routes/network/stores';
	import { writable } from 'svelte/store';
	import SchemaTree from './SchemaTree.svelte';

	export let categories: Array<string> = [];
	const dataset_description = field('dataset_description', undefined, [required()], {
		checkOnInit: true
	});
	const hint = field('hint', undefined, [required()], { checkOnInit: true });
	// const category_generator_form = form(dataset_description, hint);

	// export let cols: Array<string> = [];
	// const stored_categorization = writable(new Map());
	// $: {
	// 	if ($stored_categorization.has($dataset_description.value)) {
	// 		categories = $stored_categorization.get($dataset_description.value);
	// 	} else {
	// 		categories = [];
	// 	}
	// }

	// $: {
	// 	if (Array.from($stored_categorization.keys()).length > 0) {
	// 		cols = Array.from($stored_categorization.keys());
	// 	} else {
	// 		cols = [];
	// 	}
	// }

	// let on_keypress = async (e) => {
	// 	if (e.key === 'Enter') {
	// 		e.preventDefault();
	// 		submit_form(e);
	// 	} else {
	// 		if ($stored_categorization.has($dataset_description.value)) {
	// 			categories = $stored_categorization.get($dataset_description.value);
	// 		}
	// 	}
	// };

	// let submit_form = async (e) => {
	// 	if (!$stored_categorization.has($dataset_description.value)) {
	// 		$stored_categorization.set($dataset_description.value, []);
	// 	}
	// 	if ($dataset_description.valid) {
	// 		// let category_generation_response = await get_categories($api_url, $dataset_description.value, $hint.value);
	// 		console.log('submit_form');
	// 		let category_generation_response = await get_categorization(
	// 			$api_url,
	// 			$dataset_description.value,
	// 			categories
	// 		);
	// 		console.log(category_generation_response);
	// 		$stored_categorization
	// 			.get($dataset_description.value)
	// 			.push(...category_generation_response.categories);
	// 		$stored_categorization = $stored_categorization;
	// 	}
	// };

	// let add_category = (e) => {
	// 	if (e.key === 'Enter') {
	// 		e.preventDefault();
	// 		if ($stored_categorization.has($dataset_description.value)) {
	// 			$stored_categorization.get($dataset_description.value).push($hint.value);
	// 		} else {
	// 			$stored_categorization.set($dataset_description.value, [$hint.value]);
	// 		}
	// 		categories = $stored_categorization.get($dataset_description.value);
	// 		$stored_categorization = $stored_categorization;
	// 		$hint.value = '';
	// 	}
	// };
</script>

<SchemaTree />

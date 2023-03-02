<script lang="ts">
	import { fetchChunk } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import CheckmarkOutline from 'carbon-icons-svelte/lib/CheckmarkOutline.svelte';
	import CloseOutline from 'carbon-icons-svelte/lib/CloseOutline.svelte';
	import Help from 'carbon-icons-svelte/lib/Help.svelte';
	import { createEventDispatcher } from 'svelte';

	// Below are props (attributes) for our component.
	// These match what's in the Document class on the Python side.
	// These are Writable store objects, which means that it can be read from and written to.
	// *** To access the value of the store, use $store_name, so e.g. $data or get(store_name), so e.g. get(data)***
	export let df: DataFrameRef;

	// More component props
	export let textColumn: string;
	export let paragraphColumn: string;
	export let labelColumn: string;
	export let idColumn: string;

	let dispatch = createEventDispatcher();

	function dispatchLabel(id: string, label: number) {
		dispatch('label', {
			row_id: id,
			value: label
		});
	}

	// Fetch data for the `df` dataframe
	// This fetches all the data from the text_column
	$: textDfPromise = fetchChunk({ df, start: 0, end: null, columns: [textColumn] });

	// Fetch data for the `df` dataframe
	// This fetches all the data from the paragraph_column if it's not null
	let paragraphDfPromise: any;
	$: if (paragraphColumn) {
		paragraphDfPromise = fetchChunk({ df, start: 0, end: null, columns: [paragraphColumn] });
	}

	// Fetch data for the `df` dataframe
	// This fetches all the data from the label_column and id_column
	let labelIdDfPromise: any;
	$: if (labelColumn) {
		// The name of the id_column was told to us by the edit_target
		labelIdDfPromise = fetchChunk({
			df,
			start: 0,
			end: null,
			columns: [labelColumn, idColumn]
		});
	}

	// Here's a function that takes in an array of sentences, an array of paragraph_indices (i.e. what paragraph each sentence is in)
	// and an array containing [label, id] pairs for each sentence
	// It returns an array of objects, each of which contains the sentence, sentence_index, label and id
	let createParagraphs = (sentences: any, paragraphIndices: any, labelsAndIds: any) => {
		// Make a list of lists of paragraphs
		let paragraphs: Array<any> = [];
		let currParagraphIndex = -1;
		for (let i = 0; i < sentences.length; i++) {
			let paragraphIndex = paragraphIndices[i];
			let sentence = sentences[i];
			let [label, id] = labelsAndIds[i];
			if (paragraphIndex > currParagraphIndex) {
				paragraphs.push([]);
				currParagraphIndex = paragraphIndex;
			}
			paragraphs[currParagraphIndex].push({
				sentence: sentence,
				label: label,
				id: id
			});
		}
		return paragraphs;
	};
</script>

<div class="document">
	{#await textDfPromise}
		Waiting...
	{:then textDf}
		{#await paragraphDfPromise}
			Waiting
		{:then paragraphDf}
			{#await labelIdDfPromise}
				Waiting
			{:then labelDf}
				{#if paragraphDf}
					<!-- Create an array of paragraphs. 
					Each element is an array of objects. 
					Each object contains 'sentence', 'id', 'label' -->
					{@const paragraphs = createParagraphs(textDf.rows, paragraphDf.rows, labelDf.rows)}

					{#each paragraphs as paragraph, i}
						<div class="flex">
							<div class="font-mono pr-6 whitespace-nowrap self-center">Paragraph {i + 1}</div>
							<p>
								{#each paragraph as { sentence, label, id } (id)}
									<!-- Apply conditional styling to the sentence, depending on what the label is. -->
									<span
										{id}
										class:border-slate-700={label === -1}
										class:border-red-700={label === 0}
										class:border-emerald-700={label === 1}
										class:border-orange-700={label === 2}
										class:bg-slate-300={label === -1}
										class:bg-red-300={label === 0}
										class:bg-emerald-300={label === 1}
										class:bg-orange-300={label === 2}
										class:sentence
									>
										{sentence}
										<div class="text-interactions">
											<div class="selecting">
												<!-- svelte-ignore a11y-click-events-have-key-events -->
												<i
													class="text-red-500 rounded-full hover:bg-slate-400"
													class:bg-red-500={label === 0}
													class:text-red-100={label === 0}
													on:click={() => dispatchLabel(id, 0)}
												>
													<CloseOutline size={32} />
												</i>
												<!-- svelte-ignore a11y-click-events-have-key-events -->
												<i
													class="text-emerald-500 rounded-full hover:bg-slate-400"
													class:bg-emerald-400={label === 1}
													class:text-emerald-100={label === 1}
													on:click={() => dispatchLabel(id, 1)}
												>
													<CheckmarkOutline size={32} />
												</i>
												<!-- svelte-ignore a11y-click-events-have-key-events -->
												<i
													class="text-orange-500 rounded-full hover:bg-slate-400"
													class:bg-orange-400={label === 2}
													class:text-orange-100={label === 2}
													on:click={() => dispatchLabel(id, 2)}
												>
													<Help size={32} />
												</i>
											</div>
										</div>
									</span>
								{/each}
							</p>
						</div>
					{/each}
				{:else}
					<!-- no paragraph index -->
					{#each textDf.rows as paragraph, i}
						<div class="flex">
							<div class="font-mono pr-6 whitespace-nowrap self-center">Paragraph {i + 1}</div>
							<p>{paragraph}</p>
						</div>
					{/each}
				{/if}
			{/await}
		{/await}
	{/await}
</div>

<style>
	.document {
		@apply w-3/4 flex flex-col self-center gap-2 p-4 bg-purple-100 rounded-lg;
	}

	.sentence {
		@apply relative inline p-0 mr-0.5 bg-none text-base leading-6 text-gray-800;
		@apply border border-dotted;
	}

	.text-interactions {
		@apply relative inline-flex w-0 h-0 overflow-hidden;
	}

	.selecting {
		@apply absolute right-0 left-0 top-0 -mt-4 flex items-center z-[999] w-16 bg-gray-100;
		@apply border-r border-t border-b border-dotted border-transparent;
	}

	.selecting i {
		@apply flex justify-center items-center text-base h-5 w-5;
	}

	i::before {
		@apply flex justify-center items-center;
	}

	.sentence:hover {
		@apply text-black bg-gray-100 z-[999];
	}

	.sentence:hover > .text-interactions {
		@apply overflow-visible;
	}
</style>

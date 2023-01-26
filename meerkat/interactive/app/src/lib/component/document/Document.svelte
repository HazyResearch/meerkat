<script lang="ts">
	// Load in getContext from svelte (always)
	import { createEventDispatcher, getContext } from 'svelte';

	// Icons
	import CheckmarkOutline from 'carbon-icons-svelte/lib/CheckmarkOutline.svelte';
	import CloseOutline from 'carbon-icons-svelte/lib/CloseOutline.svelte';
	import Help from 'carbon-icons-svelte/lib/Help.svelte';

	// Running getContext('Meerkat') returns an object which contains useful functions
	// for interacting with the Python backend.
	// Each of these functions can be accessed by running $function_name
	const { fetch_chunk } = getContext('Meerkat');
	// the `get_rows` function is used to fetch data from a dataframe in the Python backend
	// the `edit` function is used to send edits to a dataframe in the Python backend

	// Below are props (attributes) for our component.
	// These match what's in the Document class on the Python side.
	// These are Writable store objects, which means that it can be read from and written to.
	// *** To access the value of the store, use $store_name, so e.g. $data or get(store_name), so e.g. get(data)***
	export let df;

	// More component props
	export let text_column: string;
	export let paragraph_column: string;
	export let label_column: string;
	export let id_column: string;

	let dispatch = createEventDispatcher();

	function dispatchLabel(id: string, label: number) {
		dispatch('label', {
			row_id: id,
			value: label
		});
	}

	// Fetch data for the `df` dataframe
	// This fetches all the data from the text_column
	$: text_df_promise = fetch_chunk(df, 0, null, null, [text_column]);

	// Fetch data for the `df` dataframe
	// This fetches all the data from the paragraph_column if it's not null
	let paragraph_df_promise: any;
	$: if (paragraph_column) {
		paragraph_df_promise = fetch_chunk(df, 0, null, null, [paragraph_column]);
	}

	// Fetch data for the `df` dataframe
	// This fetches all the data from the label_column and id_column
	let label_id_df_promise: any;
	$: if (label_column) {
		// The name of the id_column was told to us by the edit_target
		label_id_df_promise = fetch_chunk(df, 0, null, null, [label_column, id_column]);
	}

	// Here's a function that takes in an array of sentences, an array of paragraph_indices (i.e. what paragraph each sentence is in)
	// and an array containing [label, id] pairs for each sentence
	// It returns an array of objects, each of which contains the sentence, sentence_index, label and id
	let create_paragraphs = (sentences: any, paragraph_indices: any, labels_and_ids: any) => {
		// Make a list of lists of paragraphs
		let paragraphs: Array<any> = [];
		let curr_paragraph_index = -1;
		for (let i = 0; i < sentences.length; i++) {
			let paragraph_index = paragraph_indices[i];
			let sentence = sentences[i];
			let [label, id] = labels_and_ids[i];
			if (paragraph_index > curr_paragraph_index) {
				paragraphs.push([]);
				curr_paragraph_index = paragraph_index;
			}
			paragraphs[curr_paragraph_index].push({
				sentence: sentence,
				label: label,
				id: id
			});
		}
		return paragraphs;
	};
</script>

<div class="document">
	{#await text_df_promise}
		Waiting...
	{:then text_df}
		{#await paragraph_df_promise}
			Waiting
		{:then paragraph_df}
			{#await label_id_df_promise}
				Waiting
			{:then label_df}
				{#if paragraph_df}
					<!-- Create an array of paragraphs. 
					Each element is an array of objects. 
					Each object contains 'sentence', 'id', 'label' -->
					{@const paragraphs = create_paragraphs(text_df.rows, paragraph_df.rows, label_df.rows)}

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
										<div class="text_interactions">
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
					{#each text_df.rows as paragraph, i}
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

	.text_interactions {
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

	.sentence:hover > .text_interactions {
		@apply overflow-visible;
	}
</style>

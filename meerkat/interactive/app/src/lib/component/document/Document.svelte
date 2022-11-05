<script lang="ts">
	// Load in getContext from svelte (always)
	import { getContext } from 'svelte';
	// Load in the type Writable from svelte/store (always)
	import { get, type Writable } from 'svelte/store';

	// Icons
	import CheckmarkOutline from 'carbon-icons-svelte/lib/CheckmarkOutline.svelte';
	import Help from 'carbon-icons-svelte/lib/Help.svelte';
	import CloseOutline from 'carbon-icons-svelte/lib/CloseOutline.svelte';

	import type { EditTarget } from '$lib/utils/types';

	// Running getContext('Interface') returns an object which contains useful functions
	// for interacting with the Python backend.
	// Each of these functions can be accessed by running $function_name
	const { get_rows, edit } = getContext('Interface');

	// This is a prop for our component. It is a Writable store, which means that it can be
	// read from and written to.
	// *** To access the value of the store, use $store_name, so e.g. $data ***
	export let df: Writable;
	// More component props
	export let text_column: Writable<string>;
	export let paragraph_column: Writable<string>;
	export let label_column: Writable<string>;
	export let edit_target: EditTarget;

	import { createTippy } from 'svelte-tippy';
	import { followCursor } from 'tippy.js';

	let pivot_tippy = (node: HTMLElement, parameters: any = null) => {};
	pivot_tippy = createTippy({
		placement: 'auto',
		allowHTML: true,
		theme: 'pivot-tooltip',
		// followCursor: true,
		// plugins: [followCursor],
		duration: [0, 0],
		maxWidth: '95vw',
		interactive: true
	});

	$: text_df_promise = $get_rows($df.box_id, 0, null, null, [$text_column]);

	let paragraph_df_promise: any;
	$: if ($paragraph_column) {
		paragraph_df_promise = $get_rows($df.box_id, 0, null, null, [$paragraph_column]);
	}

	let label_id_df_promise: any;
	let id_column: string;
	$: if ($label_column) {
		id_column = edit_target.source_id_column;
		label_id_df_promise = $get_rows($df.box_id, 0, null, null, [$label_column, id_column]);
	}

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
				sentence_index: i,
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
					Each object contains 'sentence', 'paragraph_index', 'sentence_index' -->
					{@const paragraphs = create_paragraphs(text_df.rows, paragraph_df.rows, label_df.rows)}

					{#each paragraphs as paragraph, i}
						<div class="flex">
							<div class="font-mono pr-6 whitespace-nowrap self-center">Paragraph {i + 1}</div>
							<p>
								{#each paragraph as { sentence, sentence_index, label, id } (id)}
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
									<!-- use:pivot_tippy={{ content: document.getElementById(`${id}-pivot-tooltip`)?.innerHTML }} -->
										{sentence}
										<div class="text_interactions">
											<div class="selecting">
												<i
													class="text-red-500 rounded-full hover:bg-slate-400"
													class:bg-red-500={label === 0}
													class:text-red-100={label === 0}
													on:click={() => {
														label = 0;
														$edit(get(edit_target.target).box_id, 0, $label_column, id, id_column);
													}}
												>
													<CloseOutline size={32} />
												</i>
												<i
													class="text-emerald-500 rounded-full hover:bg-slate-400"
													class:bg-emerald-400={label === 1}
													class:text-emerald-100={label === 1}
													on:click={() => {
														label = 1;
														$edit(get(edit_target.target).box_id, 1, $label_column, id, id_column);
													}}
												>
													<CheckmarkOutline size={32} />
												</i>
												<i
													class="text-orange-500 rounded-full hover:bg-slate-400"
													class:bg-orange-400={label === 2}
													class:text-orange-100={label === 2}
													on:click={() => {
														label = 2;
														$edit(get(edit_target.target).box_id, 2, $label_column, id, id_column);
													}}
												>
													<Help size={32} />
												</i>
											</div>
										</div>
									</span>
									<!-- <div id="{id}-pivot-tooltip" class="hidden">
										<div on:click={() => {console.log("clicked.")}}>abc</div>
										<div class="selecting">
											<i
												class="text-red-500 rounded-full hover:bg-slate-400"
												class:bg-red-500={label === 0}
												class:text-red-100={label === 0}
												on:click={() => {
													label = 0;
													console.log("click");
													$edit(get(edit_target.target).box_id, 0, $label_column, id, id_column);
												}}
											>
												<CloseOutline size={32} />
											</i>
											<i
												class="text-emerald-500 rounded-full hover:bg-slate-400"
												class:bg-emerald-400={label === 1}
												class:text-emerald-100={label === 1}
												on:click={() => {
													label = 1;
													$edit(get(edit_target.target).box_id, 1, $label_column, id, id_column);
												}}
											>
												<CheckmarkOutline size={32} />
											</i>
											<i
												class="text-orange-500 rounded-full hover:bg-slate-400"
												class:bg-orange-400={label === 2}
												class:text-orange-100={label === 2}
												on:click={() => {
													label = 2;
													$edit(get(edit_target.target).box_id, 2, $label_column, id, id_column);
												}}
											>
												<Help size={32} />
											</i>
										</div>
									</div> -->
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

	:global(.tippy-box[data-theme='pivot-tooltip']) {
		@apply py-1 px-1 text-xs font-mono rounded-lg shadow-sm;
		@apply text-white bg-violet-500;
	}
</style>

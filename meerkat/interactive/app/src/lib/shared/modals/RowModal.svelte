<script lang="ts">
	import type { DataFrameRef } from '$lib/api/dataframe';
	import { getContext } from 'svelte';
	import { closeModal } from 'svelte-modals';
	import Cell from '../cell/Cell.svelte';

	export let isOpen: boolean;
	export let df: DataFrameRef;
	export let posidx: number;

	// Give the card the `flex-grow` Tailwind class to horizontally
	// fill out space in the (containing) flex container.
	export let card_flex_grow: boolean = false;
	export let as_modal: boolean = false;
	export let wrap_content: boolean = false;

	const { fetch_chunk } = getContext('Meerkat');

	$: chunk_promise = fetch_chunk({ df: df, keys: [posidx] });
    let main_column = "img";

	const onKeyPress = (e) => {
		console.log('hello');
		if (e.charCode === 27) {
			closeModal();
		}
	};
</script>

{#if isOpen}
	<div
		class="w-full fixed top-0 bottom-0 right-0 left-0 flex justify-center items-center z-[100]"
		on:keypress={onKeyPress}
	>
		<div
			class="w-[90%] bg-white rounded-lg grid-rows-2 overflow-x-hidden"
			class:flex-grow={card_flex_grow}
			class:card-modal={as_modal}
		>
			{#await chunk_promise}
				Loading...
			{:then chunk}
				<div class="grid grid-cols-[auto_1fr]">
					<div class="w-80 bg-slate-100 overflow-y-scroll drop-shadow-lg">
						<div class="flex flex-col py-3 px-5 space-y-4">
							<div class="text-center font-bold text-gray-600 text-xl">Columns</div>
							<div class="flex-col flex space-y-1 ">
								{#each chunk.columns as column}
									<button
										class="grid grid-cols-2 align-middle items-center rounded-md hover:bg-slate-200 px-3 py-1"
                                        class:bg-slate-200={main_column === column}
                                        on:click={() => {
                                            main_column = column;
                                        }}
									>
										<span class="text-bf text-gray-500 text-left">{column}</span>
										<span
											class="text-gray-700 text-right whitespace-nowrap overflow-hidden text-ellipsis"
										>
											<Cell {...chunk.get_cell(0, column)} Cell />
										</span>
									</button>
								{/each}
							</div>
						</div>
					</div>

					<div class="flex p-10 items-center justify-center justify-self-center">
						<Cell {...chunk.get_cell(0, main_column)} Cell />
					</div>
				</div>
			{/await}
		</div>
	</div>
{/if}

<script lang="ts">
	import type { DataFrameRef } from '$lib/api/dataframe';
	import { getContext } from 'svelte';
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

    const onKeyPress = (e) => {
	};
</script>

{#if isOpen}
	<div
		class="w-full fixed top-0 bottom-0 right-0 left-0 flex justify-center items-center pointer-events-none z-[100]"
	>
		<div
			class="w-[90%] bg-gray-100 rounded-lg grid-rows-2 overflow-x-hidden"
			class:flex-grow={card_flex_grow}
			class:card-modal={as_modal}
		>
			{#await chunk_promise}
				Loading...
			{:then chunk}
				<div class="grid grid-cols-[auto_1fr]">
					<div class="w-80 bg-slate-200 overflow-y-scroll">
						<div class="p-3">
							<div class="text-center text-bf text-gray-600 text-lg">Rows</div>
							<div class="flex-col flex space-y-4">
								{#each chunk.columns as column}
									<div class="grid grid-cols-2 align-middle items-center">
										<span class="text-bf text-gray-500">{column}</span>
										<span class="text-gray-700 text-ellipsis overflow-hidden justify-self-end ">
											<Cell {...chunk.get_cell(0, column)} Cell />
										</span>
									</div>
								{/each}
							</div>
						</div>
					</div>

					<div class="flex p-10 items-center justify-center justify-self-center">
						<Cell {...chunk.get_cell(0, 'img')} Cell />
					</div>
				</div>
			{/await}
		</div>
	</div>
{/if}

<style>
</style>

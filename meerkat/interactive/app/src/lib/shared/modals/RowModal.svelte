<script lang="ts">
	import type { DataFrameRef } from '$lib/api/dataframe';
	import { getContext } from 'svelte';
	import { closeModal } from 'svelte-modals';
	import Cell from '../cell/Cell.svelte';
	import ChevronLeft from 'svelte-bootstrap-icons/lib/ChevronLeft.svelte';
	import ChevronRight from 'svelte-bootstrap-icons/lib/ChevronRight.svelte';
	import ArrowLeft from 'svelte-bootstrap-icons/lib/ArrowLeft.svelte';
	import { chunk } from 'underscore';

	export let isOpen: boolean;
	export let df: DataFrameRef;
	export let posidx: number;
	export let main_column = 'img';

	// Give the card the `flex-grow` Tailwind class to horizontally
	// fill out space in the (containing) flex container.
	export let card_flex_grow: boolean = false;
	export let as_modal: boolean = false;
	export let wrap_content: boolean = false;

	const { fetch_chunk } = getContext('Meerkat');

	$: chunk_promise = fetch_chunk({ df: df, indices: [posidx] });

	const increment = async () => {
		let chunk = await chunk_promise;
		if (posidx < chunk.full_length - 1) {
			posidx += 1;
		}
	};

	const decrement = () => {
		if (posidx > 0) {
			posidx -= 1;
		}
	};

	const onKeyPress = async (e) => {
		console.log(e.charCode);
		if (e.charCode === 113) {
			closeModal();
		} else if (e.charCode === 97) {
			decrement();
		} else if (e.charCode === 100) {
			increment();
		}
	};
</script>

<svelte:window on:keypress={onKeyPress} />

{#if isOpen}
	<div class="w-full fixed top-0 bottom-0 right-0 left-0 flex justify-center items-center z-[100]">
		<div
			class="w-[90%] h-[90%] bg-white rounded-lg grid-rows-2 overflow-x-hidden"
			class:flex-grow={card_flex_grow}
			class:card-modal={as_modal}
		>
			<div class="h-full grid grid-cols-[auto_1fr]">
				<div class="w-80 bg-slate-100 overflow-y-scroll drop-shadow-lg">
					<div class="flex flex-col py-3 px-5 space-y-4">
						<div class="text-center font-bold text-gray-600 text-xl">Columns</div>
						<!-- Key-Value Pairs -->
						<div class="flex-col flex space-y-1 ">
							{#await chunk_promise then chunk}
								{#each chunk.columns as column}
									<!-- Key-Value Pair -->
									<button
										class="grid grid-cols-2 align-middle items-center rounded-md hover:bg-slate-200 px-3 py-1"
										class:bg-slate-200={main_column === column}
										on:click={() => {
											main_column = column;
										}}
									>
										<!-- Key -->
										<span
											class="text-bf text-slate-600 text-left font-mono"
											class:font-bold={main_column === column}
										>
											{column}
										</span>
										<!-- Value -->
										<span
											class="text-gray-600 text-right whitespace-nowrap overflow-hidden text-ellipsis"
										>
											<Cell {...chunk.get_cell(0, column)} Cell />
										</span>
									</button>
								{/each}
							{/await}
						</div>
					</div>
				</div>

				<div class="grid grid-rows-[auto_1fr]">
					<!-- Header section -->
					<div class="grid grid-cols-3 px-3 py-1 items-center">
						<!-- Close button -->
						<button
							class="flex items-center gap-1 text-slate-800 hover:bg-slate-100 w-fit h-fit rounded-md px-1"
							on:click={() => closeModal()}
						>
							<ArrowLeft /> Close
						</button>
						<!-- Main column header -->
						<div class="justify-self-center text-xl font-bold font-mono text-slate-600 ">
							{main_column}
						</div>
						<!-- Navigation buttons -->
						<div class="justify-self-end">
							<ul class="inline-flex self-center items-center">
								<li>
									<button
										on:click={decrement}
										class="flex items-center justify-center group w-6 h-6 rounded-lg hover:bg-slate-100 text-slate-800"
									>
										<ChevronLeft class="group-hover:stroke-2" width={16} height={16} />
									</button>
								</li>
								<li>
									{#await chunk_promise}
										<button class="w-18 px-1 h-8 text-slate-800">
											{posidx} / ?
										</button>
									{:then chunk}
										<button class="w-18 px-1 h-8 text-slate-800">
											{posidx} / {chunk.full_length}
										</button>
									{/await}
								</li>
								<li>
									<button
										on:click={increment}
										class="flex items-center justify-center group w-6 h-6 rounded-lg hover:bg-slate-100 text-slate-600"
									>
										<ChevronRight class="" width={16} height={16} />
									</button>
								</li>
							</ul>
						</div>
					</div>
					<!-- Main section -->
					<div class="flex p-10 items-center justify-center justify-self-center">
						{#await chunk_promise then chunk}
							<Cell {...chunk.get_cell(0, main_column)} Cell />
						{/await}
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}

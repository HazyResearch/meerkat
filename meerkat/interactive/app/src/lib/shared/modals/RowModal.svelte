<script lang="ts">
	import { fetchChunk, fetchSchema } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import ArrowLeft from 'svelte-bootstrap-icons/lib/ArrowLeft.svelte';
	import ChevronLeft from 'svelte-bootstrap-icons/lib/ChevronLeft.svelte';
	import ChevronRight from 'svelte-bootstrap-icons/lib/ChevronRight.svelte';
	import { closeModal } from 'svelte-modals';
	import Cell from '../cell/Cell.svelte';

	export let isOpen: boolean;
	export let df: DataFrameRef;
	export let posidx: number;
	export let mainColumn: string = '';

	// Give the card the `flex-grow` Tailwind class to horizontally
	// fill out space in the (containing) flex container.
	export let cardFlexGrow: boolean = false;
	export let asModal: boolean = false;

	$: schemaPromise = fetchSchema({ df: df, formatter: 'tiny' });
	$: chunkPromise = fetchChunk({ df: df, posidxs: [posidx], formatter: 'tag' });
	$: mainChunkPromise = schemaPromise.then((schema) => {
		if (mainColumn === '') {
			mainColumn = schema.columns[0].name;
		}
		return fetchChunk({
			df: df,
			posidxs: [posidx],
			columns: [mainColumn],
			formatter: 'full'
		});
	});

	const increment = async () => {
		let chunk = await chunkPromise;
		if (posidx < chunk.fullLength - 1) {
			posidx += 1;
		}
	};

	const decrement = () => {
		if (posidx > 0) {
			posidx -= 1;
		}
	};

	const onKeyPress = async (e) => {
		// q / esc / enter
		if (e.charCode === 113 || e.charCode === 13 || e.charCode === 27) {
			closeModal();
			// a / left arrow
		} else if (e.charCode === 97 || e.charCode === 37) {
			decrement();
			// d / right arrow
		} else if (e.charCode === 100 || e.charCode === 39) {
			increment();
		}
	};
</script>

<svelte:window on:keypress={onKeyPress} />

{#if isOpen}
	<div
		class="w-full fixed top-0 bottom-0 right-0 left-0 flex flex-col justify-center items-center z-[100]"
	>
		<div
			class="w-[90%] h-[90%] bg-white rounded-lg grid-rows-2 overflow-x-hidden"
			class:flex-grow={cardFlexGrow}
			class:card-modal={asModal}
		>
			<div class="h-full grid grid-cols-[auto_1fr]">
				<div class="w-80 bg-slate-100 overflow-y-scroll drop-shadow-lg">
					<div class="flex flex-col py-3 px-5 space-y-4">
						<div class="text-center font-bold text-gray-600 text-xl">Columns</div>
						<!-- Key-Value Pairs -->
						<div class="flex-col flex space-y-1">
							{#await schemaPromise then schema}
								{#each schema.columns as column}
									<!-- Key-Value Pair -->
									<button
										class="grid grid-cols-2 align-middle items-center rounded-md hover:bg-slate-200 px-4 py-1"
										class:bg-slate-200={mainColumn === column.name}
										on:click={() => {
											mainColumn = column.name;
										}}
									>
										<!-- Key -->
										<span
											class="text-bf text-slate-600 text-left font-mono hitespace-nowrap text-ellipsis overflow-hidden "
											class:font-bold={mainColumn === column.name}
										>
											{column.name}
										</span>
										<!-- Value -->
										<span class="text-gray-600 w-full flex justify-end">
											{#await chunkPromise then chunk}
												<Cell {...chunk.getCell(0, column.name)} />
											{/await}
										</span>
									</button>
								{/each}
							{/await}
						</div>
					</div>
				</div>

				<div class="grid grid-rows-[auto_1fr] overflow-y-scroll items-center">
					<!-- Header section -->
					<div class="grid grid-cols-3 px-3 py-1 items-center">
						<!-- Navigation buttons -->
						<div class="justify-self-start">
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
									{#await schemaPromise then schema}
										<button class="w-18 px-1 h-8 text-slate-800">
											Row <span class="font-bold">{posidx}</span> of
											<span class="font-bold">{schema.nrows}</span>
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

						<!-- Main column header -->
						<div class="justify-self-center text-xl font-bold font-mono text-slate-600 ">
							{mainColumn}
						</div>
						<!-- Close button -->
						<button
							class="flex justify-self-end items-center gap-1 text-slate-800 hover:bg-slate-100 w-fit h-fit rounded-md px-1"
							on:click={() => closeModal()}
						>
							<ArrowLeft /> Close
						</button>
					</div>
					<!-- Main section -->
					<div
						class="flex p-10 items-center h-full w-full justify-center justify-self-center overflow-y-scroll"
					>
						{#await mainChunkPromise then chunk}
							<Cell {...chunk.getCell(0, mainColumn)} />
						{/await}
					</div>
				</div>
			</div>
		</div>
		<div
			class="h-fit bg-slate-100 mt-1 w-[90%] rounded-lg text-center text-sm text-gray-600 font-mono"
		>
			a : previous row | d : next row | q : close
		</div>
	</div>
{/if}

<script lang="ts">
	import Status from '$lib/shared/common/Status.svelte';
	import { dispatch } from '$lib/utils/api';
	import type { DataFrameRef, DataFrameSchema } from '$lib/utils/dataframe';
	import type { Endpoint } from '$lib/utils/types';
	import { createEventDispatcher, tick } from 'svelte';
	import Select from 'svelte-select';
	import Textbox from '../textbox/Textbox.svelte';

	const eventDispatcher = createEventDispatcher();

	export let df: DataFrameRef;
	export let against: string;
	export let text: string;
	export let title: string = '';
	export let showAgainst: boolean = false;
	export let enableSelection: boolean = false;

	export let onMatch: Endpoint;
	export let getMatchSchema: Endpoint;

	let status: string = 'waiting';

	let schemaPromise;
	let itemsPromise;
	$: {
		schemaPromise = dispatch(getMatchSchema.endpointId, { detail: {} });
		itemsPromise = schemaPromise.then((schema: DataFrameSchema) => {
			return schema.columns.map((column) => ({ value: column.name, label: column.name }));
		});
	}

	const onKeyPress = (e: CustomEvent) => {
		console.log('onkeypress', e);
		if (e.detail.code === 'Enter') {
			console.log('matching');
			onSearch();
		} else status = 'waiting';
	};

	let onSearch = async () => {
		if (against === '') {
			status = 'error';
			return;
		}
		status = 'working';
		let promise = dispatch(onMatch.endpointId, {
			detail: { against: against, query: text }
		});
		promise
			.then(() => {
				status = 'success';
			})
			.catch((error: TypeError) => {
				status = 'error';
			});
	};

	function handleSelect(event) {
		against = event.detail.value;
	}

	function handleClear() {
		against = '';
	}
	$: againstItem = { value: against, label: against };

	let negativeClicked = false;
	let positiveClicked = false;
</script>

<div class="bg-slate-100 py-1 rounded-lg z-50 flex flex-col my-1">
	{#if title != ''}
		<div class="font-bold text-md text-slate-600 pl-2 text-center">
			{title}
		</div>
	{/if}
	<div class="form-control">
		<div class="input-group w-100% flex items-center align-middle px-3 space-x-2">
			<div class="">
				<Status {status} />
			</div>
			<Textbox bind:text on:keyup={onKeyPress} placeholder={'Write a query...'} />

			{#if enableSelection}
				<button
					class="btn btn-ghost btn-sm bg-violet-200 rounded-sm hover:bg-violet-300"
					class:bg-violet-400={negativeClicked}
					on:click={async () => {
						negativeClicked = !negativeClicked;
						// positiveClicked ? eventDispatcher('unclickplus') : null;
						// await tick();
						positiveClicked = false;
						const eventName = negativeClicked ? 'clickminus' : 'unclickminus';
						eventDispatcher(eventName);
					}}
				>
					<svg
						class="h-6 w-6"
						version="1.1"
						id="Capa_1"
						xmlns="http://www.w3.org/2000/svg"
						xmlns:xlink="http://www.w3.org/1999/xlink"
						viewBox="0 0 44.143 44.143"
						xml:space="preserve"
						fill="#000000"
						><g id="SVGRepo_bgCarrier" stroke-width="0" /><g
							id="SVGRepo_tracerCarrier"
							stroke-linecap="round"
							stroke-linejoin="round"
						/><g id="SVGRepo_iconCarrier">
							<g>
								<g>
									<path
										style="fill:#030104;"
										d="M40.727,0.177H3.416V0.176C1.532,0.176,0,1.708,0,3.592V40.55c0,1.885,1.532,3.416,3.416,3.416 h37.311c1.885,0,3.416-1.531,3.416-3.416V3.593C44.143,1.709,42.612,0.177,40.727,0.177z M42.143,40.55 c0,0.781-0.635,1.416-1.416,1.416H3.416C2.635,41.966,2,41.332,2,40.55V3.592c0-0.781,0.635-1.416,1.416-1.416h37.311 c0.781-0.001,1.416,0.634,1.416,1.416V40.55z"
									/>
									<path
										style="fill:#030104;"
										d="M34.738,21.073H9.404c-0.553,0-1,0.447-1,1s0.447,1,1,1h25.334c0.553,0,1-0.447,1-1 C35.738,21.519,35.291,21.073,34.738,21.073z"
									/>
								</g>
							</g>
						</g></svg
					>
				</button>

				<button
					class="btn btn-ghost btn-sm bg-violet-200 border border-black rounded-sm hover:bg-violet-300"
					class:bg-violet-400={positiveClicked}
					on:click={async () => {
						positiveClicked = !positiveClicked;
						// negativeClicked ? eventDispatcher('unclickminus') : null;
						// await tick();
						negativeClicked = false;
						const eventName = positiveClicked ? 'clickplus' : 'unclickplus';
						eventDispatcher(eventName);
					}}
				>
					<svg
						class="h-[22px] w-[22px] p-1"
						fill="#000000"
						viewBox="0 0 15 15"
						id="plus-16px"
						xmlns="http://www.w3.org/2000/svg"
						><g id="SVGRepo_bgCarrier" stroke-width="0" /><g
							id="SVGRepo_tracerCarrier"
							stroke-linecap="round"
							stroke-linejoin="round"
						/><g id="SVGRepo_iconCarrier">
							<path
								id="Path_60"
								data-name="Path 60"
								d="M14.5,55H8V48.5a.5.5,0,0,0-1,0V55H.5a.5.5,0,0,0,0,1H7v6.5a.5.5,0,0,0,1,0V56h6.5a.5.5,0,0,0,0-1Z"
								transform="translate(0 -48)"
							/>
						</g></svg
					>
				</button>

				<button
					class="btn btn-ghost btn-sm bg-violet-200 border border-black rounded-sm hover:bg-violet-300"
					on:click={() => {
						text = '';
						negativeClicked = false;
						positiveClicked = false;
						eventDispatcher('reset');
					}}
				>
					<svg
						class="h-[22px] w-[22px] p-[2px]"
						viewBox="0 0 24 24"
						xmlns="http://www.w3.org/2000/svg"
						fill="#000000"
						><g id="SVGRepo_bgCarrier" stroke-width="0" /><g
							id="SVGRepo_tracerCarrier"
							stroke-linecap="round"
							stroke-linejoin="round"
						/><g id="SVGRepo_iconCarrier"
							><path
								d="M22.719 12A10.719 10.719 0 0 1 1.28 12h.838a9.916 9.916 0 1 0 1.373-5H8v1H2V2h1v4.2A10.71 10.71 0 0 1 22.719 12z"
							/><path fill="none" d="M0 0h24v24H0z" /></g
						></svg
					>
				</button>

				<button
					class="btn btn-ghost btn-sm bg-violet-200 border border-black rounded-sm hover:bg-violet-300"
					on:click={() => {
						onSearch();
					}}
				>
					<svg class="h-[22px] w-[22px] p-[2px]" xmlns="http://www.w3.org/2000/svg" fill="#000000"
						><g id="SVGRepo_bgCarrier" stroke-width="0" /><g
							id="SVGRepo_tracerCarrier"
							stroke-linecap="round"
							stroke-linejoin="round"
						/><g id="SVGRepo_iconCarrier"
							><path d="M2.78 2L2 2.41v12l.78.42 9-6V8l-9-6zM3 13.48V3.35l7.6 5.07L3 13.48z" /><path
								fill-rule="evenodd"
								clip-rule="evenodd"
								d="M6 14.683l8.78-5.853V8L6 2.147V3.35l7.6 5.07L6 13.48v1.203z"
							/></g
						></svg
					>
				</button>
			{/if}

			{#if showAgainst}
				<div class="text-slate-400 px-2">against</div>

				<div class="themed pr-2 w-48">
					{#await itemsPromise}
						<Select id="column" placeholder="...a column." isWaiting={true} showIndicator={true} />
					{:then items}
						<Select
							id="column"
							placeholder="...a column."
							value={againstItem}
							{items}
							showIndicator={true}
							listPlacement="auto"
							on:select={handleSelect}
							on:clear={handleClear}
						/>
					{/await}
				</div>
			{/if}
		</div>
	</div>
</div>

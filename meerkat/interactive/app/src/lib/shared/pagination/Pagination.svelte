<script lang="ts">
	import ChevronLeft from 'svelte-bootstrap-icons/lib/ChevronLeft.svelte';
	import ChevronRight from 'svelte-bootstrap-icons/lib/ChevronRight.svelte';

	import NumberInput from './NumberInput.svelte';
	import NumberSelect from './NumberSelect.svelte';

	export let page: number = 0;
	export let per_page: number = 10;
	export let loaded_items: number;
	export let total_items: number;
	export let load_more_pages: number = 1;
	// export let loader: (start: number, end: number) => Promise<any>;
	export let load_widget: boolean = false;
	let _load_on_page_change: boolean = !load_widget;

	if (!load_widget) {
		loaded_items = total_items;
	}

	$: page_count = Math.ceil(total_items / per_page);
	$: start_item = page * per_page + 1;
	$: end_item = Math.min(page * per_page + per_page, loaded_items);

	// Functions to change page left/right
	let on_page_left = async () => {
		page = page - 1;
		if (page < 0) {
			page = 0;
		}
		// if (_load_on_page_change) {
		// 	await loader(page * per_page, (page + 1) * per_page);
		// }
	};
	let on_page_right = async () => {
		page = page + 1;
		if (page >= page_count) {
			page = page_count - 1;
		}
		// if (_load_on_page_change) {
		// 	await loader(page * per_page, (page + 1) * per_page);
		// }
	};

	let on_load_more = async () => {
		if (loaded_items >= total_items) {
			return;
		}
		let new_loaded_items = Math.min(loaded_items + per_page * load_more_pages, total_items);
		//await loader(loaded_items + 1, new_loaded_items);
		loaded_items = new_loaded_items;
	};

	let selection = { value: per_page.toString(), label: per_page };
	$: per_page = selection.label;
	$: on_change(per_page);

	async function on_change(per_page: any) {
		page = 0;
		start_item = page * per_page + 1; // why doesn't this happen automatically?!
		end_item = Math.min(page * per_page + per_page, loaded_items);
		//await loader(page * per_page, (page + 1) * per_page);
	}
</script>

<!-- <svelte:head>
	<link rel="stylesheet" href="https://unpkg.com/carbon-shared-svelte/css/g90.css" />
</svelte:head> -->

<ul class="flex justify-between py-1 px-2 bg-slate-200 rounded-xl drop-shadow-lg">
	<li class="inline-flex self-center">
		<NumberSelect
			bind:selection
			options={[
				{ value: '10', label: 10 },
				{ value: '20', label: 20 },
				{ value: '50', label: 50 },
				{ value: '100', label: 100 },
				{ value: '200', label: 200 },
				{ value: '1000', label: 1000 }
			]}
			hint_text={'Items per page'}
		/>
	</li>
	<div class="dark:text-gray-400 self-center">
		<span class="font-semibold dark:text-white">
			{start_item}-{end_item}
		</span>
		of
		<span class="font-semibold dark:text-white">
			{total_items}
		</span>
		total
	</div>
	<ul class="inline-flex self-center items-center">
		<li>
			<button
				on:click={on_page_left}
				class="flex items-center justify-center group w-6 h-6 rounded-lg bg-violet-200 text-violet-800  hover:shadow-lg "
			>
				<ChevronLeft class="group-hover:stroke-2" width={16} height={16} />
				
			</button>
		</li>
		<li>
			<button class="w-18 px-1 h-8  dark:border-gray-700  dark:bg-gray-700 dark:text-white">
				{page + 1} / {page_count}
			</button>
		</li>
		<li>
			<button
				on:click={on_page_right}
				class="flex items-center justify-center group w-6 h-6 rounded-lg bg-violet-200 text-violet-800 hover:shadow-lg "
			>
				<ChevronRight class="" width={16} height={16} />
			</button>
		</li>
	</ul>
	{#if load_widget}
		<li class="inline-flex">
			<NumberInput bind:value={load_more_pages} min={1} hint_text={'Load pages'} />
			<button
				on:click={on_load_more}
				class="inline-flex py-2 px-3 border border-solid dark:bg-gray-800 dark:border-gray-400 dark:hover:bg-gray-700"
			>
				<!-- <CloudDownload size={24} /> -->
				download
			</button>
		</li>
	{/if}
</ul>

<div class="flex justify-center ml-8">
	{#if load_widget}
		<!-- "1-10 of 32 loaded (100 total)" -->
		<div class="pt-2 dark:text-gray-400">
			<span class="font-semibold dark:text-white">
				{start_item}-{end_item}
			</span>
			of
			<span class="font-semibold dark:text-white">
				{loaded_items}
			</span>
			loaded (<span class="font-semibold dark:text-white">
				{total_items}
			</span>
			total)
		</div>
	{:else}
		<!-- "1-10 of 32 total" -->
		<!-- <div class="pt-2 dark:text-gray-400">
			<span class="font-semibold dark:text-white">
				{start_item}-{end_item}
			</span>
			of
			<span class="font-semibold dark:text-white">
				{total_items}
			</span>
			total
		</div> -->
	{/if}
</div>

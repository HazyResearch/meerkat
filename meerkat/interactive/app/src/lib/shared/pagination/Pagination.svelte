<script lang="ts">
	import { Dropdown, DropdownItem } from 'flowbite-svelte';
	import ChevronLeft from 'svelte-bootstrap-icons/lib/ChevronLeft.svelte';
	import ChevronRight from 'svelte-bootstrap-icons/lib/ChevronRight.svelte';

	export let page: number = 0;
	export let perPage: number = 10;
	export let totalItems: number;
	export let dropdownPlacement: string = 'bottom';
	export let allowSetPerPage: boolean = true;

	$: pageCount = Math.ceil(totalItems / perPage);
	$: startItem = page * perPage + 1;
	$: endItem = Math.min(page * perPage + perPage, totalItems);

	// Functions to change page left/right
	let onPageLeft = async () => {
		page = page - 1;
		if (page < 0) {
			page = 0;
		}
	};
	let onPageRight = async () => {
		page = page + 1;
		if (page >= pageCount) {
			page = pageCount - 1;
		}
	};

	let selection = { value: perPage.toString(), label: perPage };
	$: perPage = selection.label;
	$: onChange(perPage);

	async function onChange(per_page: any) {
		// page = 0;
		startItem = page * per_page + 1; // why doesn't this happen automatically?!
		endItem = Math.min(page * per_page + per_page, totalItems);
	}

	// Page size variable
	let open: boolean = false;
</script>

<ul class="inline-flex self-center items-center">
	<li>
		<button
			on:click={onPageLeft}
			class="flex items-center justify-center group w-6 h-6 rounded-lg hover:bg-slate-200 text-slate-600"
		>
			<ChevronLeft class="group-hover:stroke-2" width={16} height={16} />
		</button>
	</li>
	<li>
		<button
			class="flex w-18 px-1 h-8 text-slate-600 items-center gap-1"
			on:click={() => {
				open = !open;
			}}
		>
			{#if pageCount > 0}
				Page <span class="font-bold">{page + 1}</span> of
				<span class="font-bold">{pageCount}</span>
			{:else}
				No pages
			{/if}
		</button>
		{#if allowSetPerPage}
			<Dropdown placement={dropdownPlacement} {open} class="w-fit">
				{#each [10, 20, 50, 100, 200] as p}
					<DropdownItem
						on:click={() => {
							open = false;
							perPage = p;
						}}
					>
						<div class="text-slate-600">
							Page size <span class="font-bold">{p}</span>
						</div>
					</DropdownItem>
				{/each}
			</Dropdown>
		{/if}
	</li>
	<li>
		<button
			on:click={onPageRight}
			class="flex items-center justify-center group w-6 h-6 rounded-lg hover:bg-slate-200 text-slate-600"
		>
			<ChevronRight class="" width={16} height={16} />
		</button>
	</li>
</ul>

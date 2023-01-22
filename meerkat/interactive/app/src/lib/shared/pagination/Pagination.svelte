<script lang="ts">
	import ChevronLeft from 'svelte-bootstrap-icons/lib/ChevronLeft.svelte';
	import ChevronRight from 'svelte-bootstrap-icons/lib/ChevronRight.svelte';
	import { Dropdown, DropdownItem } from 'flowbite-svelte';
	import { tippy } from 'svelte-tippy';

	export let page: number = 0;
	export let per_page: number = 10;
	export let total_items: number;

	$: page_count = Math.ceil(total_items / per_page);
	$: start_item = page * per_page + 1;
	$: end_item = Math.min(page * per_page + per_page, total_items);

	// Functions to change page left/right
	let on_page_left = async () => {
		page = page - 1;
		if (page < 0) {
			page = 0;
		}
	};
	let on_page_right = async () => {
		page = page + 1;
		if (page >= page_count) {
			page = page_count - 1;
		}
	};

	let selection = { value: per_page.toString(), label: per_page };
	$: per_page = selection.label;
	$: on_change(per_page);

	async function on_change(per_page: any) {
		page = 0;
		start_item = page * per_page + 1; // why doesn't this happen automatically?!
		end_item = Math.min(page * per_page + per_page, total_items);
	}

	// Page size variable
	let open: boolean = false;
</script>

<ul class="inline-flex self-center items-center">
	<li>
		<button
			on:click={on_page_left}
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
			Page <span class="font-bold">{page + 1}</span> of
			<span class="font-bold">{page_count}</span>
		</button>
		<Dropdown {open} class="w-fit">
			{#each [10, 20, 50, 100, 200] as p}
				<DropdownItem
					on:click={() => {
						open = false;
						per_page = p;
					}}
				>
					<div class="text-slate-600">
						Page size <span class="font-bold">{p}</span>
					</div>
				</DropdownItem>
			{/each}
		</Dropdown>
	</li>
	<li>
		<button
			on:click={on_page_right}
			class="flex items-center justify-center group w-6 h-6 rounded-lg hover:bg-slate-200 text-slate-600"
		>
			<ChevronRight class="" width={16} height={16} />
		</button>
	</li>
</ul>

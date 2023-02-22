<script lang="ts">
	import RowModal from '$lib/shared/modals/RowModal.svelte';
	import Pagination from '$lib/shared/pagination/Pagination.svelte';
	import { fetchChunk, fetchSchema } from '$lib/utils/api';
	import type { DataFrameRef } from '$lib/utils/dataframe';
	import { Dropdown, DropdownItem } from 'flowbite-svelte';
	import { setContext, getContext } from 'svelte';
	import { BarLoader } from 'svelte-loading-spinners';
	import { openModal } from 'svelte-modals';

	export let df: DataFrameRef;
	export let selected: Array<string>;

	export let page: number = 0;
	export let perPage: number = 20;
	export let cellSize: number = 24;

	export let allowSelection: boolean = false;

	const componentId = getContext("componentId");

	$: schemaPromise = fetchSchema({
		df: df,
		formatter: 'small'
	});

	setContext('open_row_modal', (posidx: number) => {
		openModal(RowModal, {
			df: df,
			posidx: posidx,
			mainColumn: undefined
		});
	});

	$: chunkPromise = fetchChunk({
		df: df,
		start: page * perPage,
		end: (page + 1) * perPage,
	});

	let dropdownOpen: boolean = false;
</script>
TABLE

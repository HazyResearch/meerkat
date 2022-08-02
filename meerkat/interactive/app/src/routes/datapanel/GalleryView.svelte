<script lang="ts">
	export let columns: Array<string> = [];
	export let types: Array<string> = [];
	// Columns to use for tags in the GalleryView.
	export let tagColumns: Array<string> = [];
	// Main column to display.
    export let mainColumn: string = 'image';
	export let rows: Array<any> = [];
	// Total number of examples to show per page.
	export let nshow: number = 5;
	// Number of panels to display per page.
    export let npanels: number = 4;

	let nrows: number = rows.length;
	let npages: number = Math.ceil(nrows / (nshow * npanels));
	let page: number = 0;

	let tagIndices: Array<number> = tagColumns.map(tag => columns.indexOf(tag));
	let mainIndex: number = columns.indexOf(mainColumn);
	let mainType: string = types[mainIndex];
	
	// 5% saved for padding
	let tagWidth = 95 / tagIndices.length + "%";

	let value_formatter = (value: any, type: string) => {
		if (type === 'string') {
			return value;
		} else if (type === 'number') {
			return value.toFixed(2);
		} else if (type === 'image') {
			return `<img class="block object-center h-4/5 w-full rounded-md shadow-lg" src="${value}"/>`;
		} else {
			return value;
		}
	};
</script>

<!-- <svelte:head>
	<title>DataPanel Table View</title>
</svelte:head> -->

<style>
	.panel {
		display: flex;
		flex: row;
		overflow-x: auto;
		position: relative
	}

	.card {
		margin-right: 5px;
		margin-left: 5px;
		width: 100%; 
  		height: auto; 
		/* width: auto;
  		height:100px; */
	}

	.tag {
		height: 10%;
		width: 40%;
		border-radius: 7px;
		background-color: #e0e0e0;
		border-width: 1.3px;
		border-style: transparent;
		padding-left: 10px;
		padding-right: 10px;
		padding-top: 3px;
		padding-bottom: 3px;
		margin-top: 5px;
		margin-bottom: 5px;
		margin-right: 10px;
		font-size: 10px;
		font-weight: bold;
		color: #666666;
		display: inline-block;
		user-select: none;
		vertical-align: 55%;
		transition: all 0.2s ease-in-out;
		text-align: center;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.tag:hover {
		background-color: #c0c0c0;
	}

	.tooltip {
		/* visibility: hidden; */
		display: none;
		position: absolute;
		width: 20%;
		background-color: #fff;
		border-color: black;
		border-width: 1.5px;
		border-radius: 5px;
		padding: 2px;
		opacity: 0;
		transition: opacity 0.5s ease;
		text-align: center;
	}

	.tag:hover + .tooltip {
		/*visibility: visible; */
		display: initial;
		transition: opacity 0.5s ease;
		opacity: 1;
	}


	.tag-panel {
		display: flex;
		flex-direction: row;
		flex-wrap: wrap;
		justify-content: center;
		align-items: center;
		margin-top: 10px;
		margin-bottom: 10px;
	}


</style>

<div class="border border-dotted border-gray-100 bg-slate-800 p-2">
	<div class="flex flex-row justify-end w-full space-x-2">
		<!-- Buttons for switching between different table views. -->
		<button class="bg-slate-500 hover:bg-slate-700 rounded font-bold px-4">Table</button>
		<button class="bg-slate-500 hover:bg-slate-700 rounded font-bold px-4">Gallery</button>
	</div>

	{#each [...Array(npanels).keys()] as panelId}
		<div class="panel m-3 relative">
			{#each rows.slice(page * (nshow * npanels) + (panelId * nshow), page * (nshow * npanels) + ((panelId + 1) * nshow)) as row}
				<div class="card max-w-sm bg-white rounded-lg border border-gray-200 shadow-md dark:bg-gray-800 dark:border-gray-700">
					<div> {@html value_formatter(row[mainIndex], mainType)} </div>
					<div class="tag-panel"> 
						{#each tagIndices as tagIndex}
						
						<div class="tag">
							{@html value_formatter(row[tagIndex], types[tagIndex])}
						</div>
						<div class='tooltip'>{@html value_formatter(row[tagIndex], types[tagIndex])}</div>
						{/each}
					</div>
				</div>
			{/each}
		</div>
	{/each}


	<nav class="flex justify-between items-center pt-4" aria-label="Table navigation">
		<span class="text-sm font-normal text-gray-500 dark:text-gray-400">
			Showing
			<span class="font-semibold text-gray-900 dark:text-white"
				>{page * nshow + 1}-{Math.min(page * nshow + nshow, nrows)}</span
			>
			of
			<span class="font-semibold text-gray-900 dark:text-white">{nrows}</span>
		</span>
		<ul class="inline-flex items-center -space-x-px">
			<li>
				<button
					on:click={() => {
						if (page > 0) {
							page = page - 1;
						}
					}}
					class="block py-2 px-3 ml-0 leading-tight text-gray-500 bg-white rounded-l-lg border border-gray-300 hover:bg-gray-100 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
				>
					<span class="sr-only">Previous</span>
					<svg
						class="w-5 h-5"
						aria-hidden="true"
						fill="currentColor"
						viewBox="0 0 20 20"
						xmlns="http://www.w3.org/2000/svg"
						><path
							fill-rule="evenodd"
							d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z"
							clip-rule="evenodd"
						/></svg
					>
				</button>
			</li>
			<li>
				<button
					aria-current="page"
					class="z-10 py-2 px-3 leading-tight text-blue-600 bg-blue-50 border border-blue-300 hover:bg-blue-100 hover:text-blue-700 dark:border-gray-700 dark:bg-gray-700 dark:text-white"
					>{page + 1}</button
				>
			</li>
			<li>
				<button
					on:click={() => {
						if (page < npages - 1) {
							page = page + 1;
						}
					}}
					class="block py-2 px-3 leading-tight text-gray-500 bg-white rounded-r-lg border border-gray-300 hover:bg-gray-100 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
				>
					<span class="sr-only">Next</span>
					<svg
						class="w-5 h-5"
						aria-hidden="true"
						fill="currentColor"
						viewBox="0 0 20 20"
						xmlns="http://www.w3.org/2000/svg"
						><path
							fill-rule="evenodd"
							d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
							clip-rule="evenodd"
						/></svg
					>
				</button>
			</li>
		</ul>
	</nav>
</div>

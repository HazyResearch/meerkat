<script lang="ts">
    import _, { zip } from 'underscore';

	export let columns: Array<string> = [];
	export let types: Array<string> = [];
	export let rows: Array<any> = [];
	export let nshow: number = 10;

	async function fetch_url(url: string): Promise<any> {
		const res: Response = await fetch(url);
		if (!res.ok) {
			throw new Error('HTTP status ' + res.status);
		}
		const json = await res.json();
		return json;
	}

	const url = `http://localhost:7860/data_id`;
	let data_promise = fetch_url(url).then(json => json);

	let nrows: number = rows.length;
	let npages: number = Math.ceil(nrows / nshow);
	let page: number = 0;
    let sorted_by: {col_index: number, ascending: boolean} = {'col_index': -1, 'ascending': true};

	let value_formatter = (value: any, type: string) => {
		if (type === 'string') {
			return value;
		} else if (type === 'number') {
            return value;
		} else if (type === 'image') {
			return `<img class="block object-center h-4/5 w-full rounded-md shadow-lg" src="${value}"/>`;
		} else {
			return value;
		}
	};

    let sort = (col_index: number) => {
        let type = types[col_index];
        let ascending = sorted_by['col_index'] === col_index ? !sorted_by['ascending'] : true;
        sorted_by = {'col_index': col_index, 'ascending': ascending};
        let sorted_rows = _.sortBy(rows, (row: any) => {
            let value = row[col_index];
            if (type === 'number') {
                return value;
            } else {
                return value.toLowerCase();
            }
        });
        if (!ascending) {
            sorted_rows = sorted_rows.reverse();
        }
        rows = sorted_rows;
    };

</script>

<!-- <svelte:head>
	<title>DataPanel Table View</title>
</svelte:head> -->


<div class="border border-dotted border-gray-100 bg-slate-800 p-2">
    
    <div class="m-3 overflow-x-auto relative">
        <table class="w-full text-sm text-left text-gray-500 dark:text-gray-400">
            <thead class="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
                <tr>
                    <th scope="col" class="py-2 px-4 resize-x [overflow:auto]">
                        <div class="flex items-center">A
                        <button>
                            <svg xmlns="http://www.w3.org/2000/svg" class="ml-1 w-3 h-3" aria-hidden="true" fill="currentColor" viewBox="0 0 320 512"><path d="M27.66 224h264.7c24.6 0 36.89-29.78 19.54-47.12l-132.3-136.8c-5.406-5.406-12.47-8.107-19.53-8.107c-7.055 0-14.09 2.701-19.45 8.107L8.119 176.9C-9.229 194.2 3.055 224 27.66 224zM292.3 288H27.66c-24.6 0-36.89 29.77-19.54 47.12l132.5 136.8C145.9 477.3 152.1 480 160 480c7.053 0 14.12-2.703 19.53-8.109l132.3-136.8C329.2 317.8 316.9 288 292.3 288z"/></svg>
                        </button>
                    </div></th>
                    <th scope="col" class="py-2 px-4 resize-x [overflow:auto]"><div class="flex items-center">B
                        <button>
                            <svg xmlns="http://www.w3.org/2000/svg" class="ml-1 w-3 h-3" aria-hidden="true" fill="currentColor" viewBox="0 0 320 512"><path d="M27.66 224h264.7c24.6 0 36.89-29.78 19.54-47.12l-132.3-136.8c-5.406-5.406-12.47-8.107-19.53-8.107c-7.055 0-14.09 2.701-19.45 8.107L8.119 176.9C-9.229 194.2 3.055 224 27.66 224zM292.3 288H27.66c-24.6 0-36.89 29.77-19.54 47.12l132.5 136.8C145.9 477.3 152.1 480 160 480c7.053 0 14.12-2.703 19.53-8.109l132.3-136.8C329.2 317.8 316.9 288 292.3 288z"/></svg>
                        </button>
                    </div></th>
                    <th scope="col" class="py-2 px-4 resize-x [overflow:auto]"><div class="flex items-center">B
                        <button>
                            <svg xmlns="http://www.w3.org/2000/svg" class="ml-1 w-3 h-3" aria-hidden="true" fill="currentColor" viewBox="0 0 320 512"><path d="M27.66 224h264.7c24.6 0 36.89-29.78 19.54-47.12l-132.3-136.8c-5.406-5.406-12.47-8.107-19.53-8.107c-7.055 0-14.09 2.701-19.45 8.107L8.119 176.9C-9.229 194.2 3.055 224 27.66 224zM292.3 288H27.66c-24.6 0-36.89 29.77-19.54 47.12l132.5 136.8C145.9 477.3 152.1 480 160 480c7.053 0 14.12-2.703 19.53-8.109l132.3-136.8C329.2 317.8 316.9 288 292.3 288z"/></svg>
                        </button>
                    </div></th>
                </tr>
            </thead>
            <tbody>
                <tr class="bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600">
                    <td>a quick brown fox jumped over the lazy dog and it went </td>
                    <td>a quick brown fox jumped over the lazy dog and it went </td>
                    <td>b</td>
                </tr>
                <tr class="bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600">
                    <td>a quick brown fox jumped over</td>
                    <td>b</td>
                    <td>b</td>
                </tr>
            </tbody>
        </table>
    </div>

	<div class="flex flex-row justify-end w-full space-x-2">
		<!-- Buttons for switching between different table views. -->
		<button class="bg-slate-500 hover:bg-slate-700 rounded font-bold px-4">Table</button>
		<button class="bg-slate-500 hover:bg-slate-700 rounded font-bold px-4">Gallery</button>
	</div>
	<div>
		{#await data_promise}
			<div>Loading data...</div>
		{:then data}
			<div>{data}</div>
		{/await}
	</div>

	<div class="m-3 overflow-x-auto relative">
        <!-- table-fixed is necessary otherwise the columns cannot be resized when the table overflows horizontally -->
		<table class="table-fixed w-full text-sm text-left text-gray-500 dark:text-gray-400">
			<thead class="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
				<tr>
					{#each columns as column}
						<!-- <th class="border bg-slate-500 hover:bg-slate-700">{column}</th> -->
						<!-- <th scope="col" class="py-2 px-4">{column}</th> -->
						<th scope="col" class="py-2 px-4">
							<div class="flex items-center">
								{column}
								<button on:click={() => {}}>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										class="ml-1 w-3 h-3"
										aria-hidden="true"
										fill="currentColor"
										viewBox="0 0 320 512"
										><path
											d="M27.66 224h264.7c24.6 0 36.89-29.78 19.54-47.12l-132.3-136.8c-5.406-5.406-12.47-8.107-19.53-8.107c-7.055 0-14.09 2.701-19.45 8.107L8.119 176.9C-9.229 194.2 3.055 224 27.66 224zM292.3 288H27.66c-24.6 0-36.89 29.77-19.54 47.12l132.5 136.8C145.9 477.3 152.1 480 160 480c7.053 0 14.12-2.703 19.53-8.109l132.3-136.8C329.2 317.8 316.9 288 292.3 288z"
										/></svg
									>
								</button>
							</div>
						</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each rows.slice(page * nshow, (page + 1) * nshow) as row}
					<tr
						class="bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600"
					>
						{#each zip(row, types) as [value, type]}
							<td class="py-2 px-4 overflow-auto break-words">
                                {@html value_formatter(value, type)}
                            </td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	</div>

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


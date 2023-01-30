<script lang="ts">
	import Instance from './Instance.svelte';

	import Description from './Description.svelte';
	import Matrix, { type MatrixType } from './matrix/Matrix.svelte';
	import Stats from './Stats.svelte';
	import type { StatsType, InstanceType } from './types';

	export let name: string;
	export let descriptions: Array<{ score: number; description: string; }>;
	export let stats: StatsType;
	export let instances: Array<InstanceType>;
	export let plot: { PLOT_TYPE: string; matrix: undefined|MatrixType; html: string; };

	let offset: number = 10;
	let limit: number = 10;

	async function fetch_url (url: string): Promise<any> {
		const res: Response = await fetch(url);
		if (!res.ok) {
			throw new Error("HTTP status " + res.status);
		}
		const json = await res.json();
		return json;
	};

	export let loadmore = async function() {
		let moredata_promise = fetch_url('http://localhost:8900/imagenette?slice=' + 0 + '&offset=' + offset);
		offset += limit;
		let moredata = await moredata_promise;
		instances = instances.concat(moredata['slices'][0].instances);
	};
</script>

<div class="pt-2 px-6 pb-2 h-fit mx-auto bg-white rounded-xl shadow-lg overflow-x-hidden ml-4">
	<div class="h-10 mb-2 flex space-x-6 items-center">
		<div class="whitespace-nowrap text-xl font-medium text-black">
			{name}
		</div>
		<div class="mx-auto flex h-full items-center space-x-4 overflow-x-scroll no-scrollbar ml-4">
			{#each descriptions as description}
				<Description score={description.score} description={description.description} />
			{/each}
		</div>
	</div>
	<div class="flex h-40">
		<Stats stats={stats} />

		{#if plot.PLOT_TYPE === 'matrix' && plot.matrix}
			<Matrix {...plot.matrix} />
		{:else if plot.PLOT_TYPE === 'plotly'}
			{plot.html} 
		{/if}

		<div class="mx-auto flex h-full items-center space-x-4 overflow-x-scroll no-scrollbar ml-4">
			{#each instances as instance}
				<Instance {...instance} />
			{/each}
			<div>
				<!-- Load More Button -->
				<button on:click={loadmore} class="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded-lg">
					Load More
				</button>
			</div>
		</div>
	</div>
</div>

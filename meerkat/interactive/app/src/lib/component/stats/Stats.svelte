<script lang="ts">
	import { type Writable } from 'svelte/store';
	import { get } from 'svelte/store';
	import RadialProgress from './RadialProgress.svelte';

	export let data: any;
	export let specs: any;
	for (let [k, v] of Object.entries(data)) {

		v.subscribe(() => {
            console.log("new value", v)
			data = data;
		});
	}

</script>
<div class="m-2 flex flex-wrap justify-center gap-x-2 gap-y-2">
	{#each Object.entries(data) as [k, v]}
		<div class="bg-slate-100 rounded-md flex flex-col shadow-sm">
			<div class="text-slate-400 px-3 py-1 self-center">{k}</div>
			<div class="font-bold text-2xl self-center ">
				{get(v).toFixed(2)}
			</div>
		</div>
	{/each}
</div>

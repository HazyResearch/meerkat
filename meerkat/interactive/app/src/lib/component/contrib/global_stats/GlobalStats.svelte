<script lang="ts">
	import Stats from '$lib/component/core/stats/Stats.svelte';
	import { ArrowRight } from 'svelte-bootstrap-icons';
    import { InfoCircle } from 'svelte-bootstrap-icons';


	export let v1_name: string;
	export let v2_name: string;
	export let v1_mean: number;
	export let v2_mean: number;
	export let shift: number;
	export let inconsistency: number;
	export let metric: string;

    const shift_name = `${metric} Shift`;
    let data: any = {};
    data[`${metric} Shift`] = shift;
    data[`${metric} Inconsistency`] = inconsistency;

</script>

<div class="flex flex-col gap-2">
    <div class="px-3 py-1 bg-slate-100 rounded-md flex gap-4 items-center">
        <InfoCircle  class="text-violet-800"/>
        <div class="text-slate-800 text-center text-left text-sm">
            This <span class="font-mono">ChangeList</span> compares the performance of model versions:
        </div>
    </div>

	<div class="grid grid-cols-[1fr_auto_1fr] justify-items-center items-center">
        <div class="flex flex-col gap-1 items-center">
		    <div class="font-bold text-slate-800 text-md rounded-sm bg-slate-100 px-2">{v1_name}</div>
            <div class="w-fit text-slate-800 text-md rounded-sm font-mono">v1 {metric}: {v1_mean.toFixed(3)}</div>
        </div>
		<ArrowRight/>
        <div class="flex flex-col gap-1 items-center">
		    <div class="font-bold text-slate-800 text-md rounded-sm bg-slate-100 px-2">{v2_name}</div>
            <div class="w-fit text-slate-800 text-md rounded-sm font-mono">v2 {metric}: {v2_mean.toFixed(3)}</div>
        </div>	</div>
	<Stats data={data} />
</div>

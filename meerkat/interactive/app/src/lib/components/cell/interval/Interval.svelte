<script lang="ts">
	import BasicCell from '../basic/Basic.svelte';

	export let data: Array<number>;
	export let dtype: string | undefined = undefined;
	export let precision: number = 3;
	export let percentage: boolean = false;
</script>

<div class="flex flex-col justify-center">
	<BasicCell data={data[0]} {dtype} {precision} {percentage} />
    <div class="flex">
        <BasicCell data={data[0]} {dtype} {precision} {percentage} />
        <BasicCell data={data[1]} {dtype} {precision} {percentage} />
    </div>
</div>

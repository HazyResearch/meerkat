<script lang="ts">
    import Icon from "$lib/component/core/icon/Icon.svelte";
	export let data: any;
	export let dtype: string;
</script>

<div class="flex gap-1 rounded-md px-1 bg-slate-200 shadow-sm items-center my-1">
    <Icon name={"BoxFill"}/>
	<div class="text-xs flex flex-col">
        <span class="text-violet-600 font-mono font-bold">{dtype}</span>
		 <span class="font-mono">[{data.shape}]</span>
	</div>
</div>

<script lang="ts">
	export let title: string;
	export let byline: string | null = null;
	export let description: string | null = null;
	export let tight: boolean;
</script>

<div class="flex flex-col flex-wrap {tight ? `p-4` : `p-6`} bg-slate-50 rounded-lg shadow-lg dark:bg-gray-800">
	<slot name="header" />
	<div>
		<h4 class="mb-1 text-lg font-semibold text-gray-800 dark:text-white">{title}</h4>
		{#if byline}
			<h4 class="mb-1 text-sm text-gray-400 dark:text-white">{byline}</h4>
		{/if}
		
        <slot name="description">
			{#if description}
				<p class="text-sm text-gray-600 dark:text-gray-400">
					{description}
				</p>
			{/if}
        </slot>
	</div>
</div>

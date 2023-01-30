<script lang="ts">
	export let input: string;
	export let tags: Record<string, string>;
	export let correct: Record<string, boolean>;
</script>

<div class="h-full min-w-max">
	<img
		class="block object-center h-4/5 w-full rounded-md shadow-lg"
		src={input}
		alt="Alt"
	/>
	<div class="flex mt-2 space-x-2 h-max">
		{#each Object.entries(tags) as key, value}
			<span
				class="inline-block h-fit bg-gray-200 rounded-lg p-1 text-xs text-gray-700 whitespace-nowrap"
			>
				{key}
				{value}
			</span>
		{/each}

		{#if correct}
			{#each Object.entries(correct) as key, correct}
				<span
					class="inline-block h-fit bg-gray-200 rounded-lg p-1 text-xs text-gray-700 whitespace-nowrap"
				>
					{#if correct}
						{key} <span class="text-green-500">✓</span>
					{:else}
						{key} <span class="text-red-500">✗</span>
					{/if}
					{key} <u />
				</span>
			{/each}
		{/if}
	</div>
</div>

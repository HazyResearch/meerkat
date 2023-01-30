<script lang="ts">
	import type { StatsType } from "./types";
	export let stats: StatsType;
</script>

<table class="table-auto h-1 border border-separate border-spacing-2">
	<tbody>
		{#each Object.entries(stats) as key, value}
			<tr class="border-spacing-0">
				<th>
					<p class="text-left text-sm text-gray-500 whitespace-nowrap">{key}</p>
				</th>
				<td>
					<p class="bg-gray-200 rounded-lg px-3 py-1 text-sm font-semibold text-gray-700 w-fit">
						{value}
					</p>
				</td>
			</tr>
		{/each}
	</tbody>
</table>

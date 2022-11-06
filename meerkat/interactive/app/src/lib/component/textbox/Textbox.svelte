<script lang="ts">
	import { type Writable } from 'svelte/store';

	export let text: Writable<string>;
	export let title: string = '';
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md flex flex-col">
	{#if title != ''}
		<div class="font-bold text-xl text-slate-600 self-start pl-2">
			{title}
		</div>
	{/if}
	<div class="form-control">
		<div class="input-group w-100% flex items-center">
			<input
				type="text"
				placeholder="Write some text..."
				class="input input-bordered grow h-10 px-3 rounded-md shadow-md"
				on:keypress={(e) => {
					if (e.charCode == 13) {
						$text = e.target.value;
					}
				}}
			/>
		</div>
	</div>
</div>

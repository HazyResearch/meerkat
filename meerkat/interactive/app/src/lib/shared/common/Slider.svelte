<script lang="ts">
	export let id: string = 'range';
	export let min: number;
	export let max: number;
	export let value: number;
	export let step: number;
	export let size: 'small' | 'large';

	let inputClass: string =
		' w-full bg-slate-200 rounded-lg cursor-pointer self-center accent-violet-600';

</script>

<input
	{...$$restProps}
	{id}
	name={id}
	type="range"
	{min}
	{max}
	bind:value
	{step}
	{size}
	class={inputClass}
	on:change
	on:click
	on:keydown
	on:keypress
	on:keyup
/>
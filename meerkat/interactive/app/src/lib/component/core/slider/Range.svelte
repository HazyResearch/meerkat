<script lang="ts">
	export let value: number;
    export let classes: string = "";
</script>

<input
	type="range"
	bind:value
	{...$$restProps}
	class={classes}
	on:change
	on:click
	on:keydown
	on:keypress
	on:keyup
/>

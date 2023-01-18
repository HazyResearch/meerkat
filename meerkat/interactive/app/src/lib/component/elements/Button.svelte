<script lang="ts">
	export let classes: string = '';
</script>

<button
    type="button"
	on:blur
	on:click
	on:contextmenu
	on:dblclick
	on:focus
	on:mousedown
	on:mouseenter
	on:mouseleave
	on:mousemove
	on:mouseout
	on:mouseover
	on:mouseup
	class={classes}
>
	<slot />
</button>

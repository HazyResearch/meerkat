<svelte:options accessors={true} />

<script>
	import { createEventDispatcher } from 'svelte';

	let shown = false;

	let dispatch = createEventDispatcher();

	export function show() {
		shown = !shown;
		dispatch('show', shown);
	}
</script>

{#if shown}
	<slot />
{/if}


<!-- Use the following snippet to condition on the `shown` prop of this component in a parent component. -->
<!-- {#if hidden_element && hidden_element.shown}
	<div>Hidden child is shown! Put whatever content in here.</div>
{/if} -->
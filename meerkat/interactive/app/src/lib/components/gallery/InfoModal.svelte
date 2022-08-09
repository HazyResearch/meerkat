<script lang="ts">
	import { closeModal } from 'svelte-modals';
	import InfoCard from './InfoCard.svelte';
	export let is_open: boolean;
	export let props: any;
</script>

{#if is_open}
	<div
		class="fixed top-0 bottom-0 right-0 left-0 flex justify-center items-center pointer-events-none z-10"
		on:click={closeModal}
	>
		<div class="pointer-events-auto">
			<InfoCard as_modal={true} {...props} />
		</div>
	</div>
{/if}

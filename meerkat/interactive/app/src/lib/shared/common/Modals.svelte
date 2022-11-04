<script>
	import { closeModal, Modals } from 'svelte-modals';
</script>

<Modals>
	<div slot="backdrop" class="fixed top-0 bottom-0 left-0 right-0 bg-black opacity-50" on:click={closeModal} />
</Modals>

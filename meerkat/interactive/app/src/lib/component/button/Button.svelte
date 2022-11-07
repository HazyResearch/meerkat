<script lang="ts">
	import { type Writable } from 'svelte/store';
    import { writable } from 'svelte/store';

	export let value: Writable;
	export let title: string = '';
</script>

<div class="bg-slate-100 py-3 rounded-lg drop-shadow-md flex flex-col">
    <button on:click={(e) => {console.log(e); $value = {'type': e.type, 'detail': e.detail}}}>{title}</button>
</div>
